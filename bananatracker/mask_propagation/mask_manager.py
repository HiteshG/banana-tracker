"""MaskManager: SAM2.1 + Cutie for mask creation and temporal propagation.

Architecture:
  - SAM2.1 (HuggingFace) -> Creates initial masks from bounding boxes (one-time per new tracklet)
  - Cutie -> Propagates those masks temporally across all subsequent frames

The masks are used to enrich the cost matrix in track association, improving
tracking robustness especially when players are close together.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# Constants for mask creation
OVERLAP_MEASURE_VARIANT = 1
OVERLAP_VARIANT_2_GRID_STEP = 10
MASK_CREATION_BBOX_OVERLAP_THRESHOLD = 0.6


class MaskManager:
    """Manages mask creation (SAM2.1) and temporal propagation (Cutie).

    Workflow:
    1. First frame: SAM2.1 creates masks for all detected tracklets
    2. Subsequent frames:
       - Cutie propagates existing masks to current frame
       - SAM2.1 creates masks for any new tracklets
       - Removed tracklets have their masks purged from Cutie memory
    """

    def __init__(
        self,
        sam2_model_id: str = "facebook/sam2.1-hiera-large",
        sam2_checkpoint: Optional[str] = None,
        cutie_weights_path: Optional[str] = None,
        device: str = "cuda:0",
        hf_token: Optional[str] = None,
        mask_start_frame: int = 1,
        bbox_overlap_threshold: float = 0.6
    ):
        """Initialize SAM2.1 and Cutie models.

        Parameters
        ----------
        sam2_model_id : str
            HuggingFace model ID for SAM2.1 (default: facebook/sam2.1-hiera-large)
        sam2_checkpoint : str, optional
            Local checkpoint path for SAM2.1 (overrides HF download)
        cutie_weights_path : str, optional
            Path to Cutie weights file (cutie-base-mega.pth)
        device : str
            Device for inference (cuda:0, cpu, etc.)
        hf_token : str, optional
            HuggingFace token for gated model access
        mask_start_frame : int
            Frame number to start mask processing
        bbox_overlap_threshold : float
            Threshold for avoiding overlapped mask creation
        """
        self.device = device
        self.SAM_START_FRAME = mask_start_frame
        self.bbox_overlap_threshold = bbox_overlap_threshold

        # State tracking
        self.masks = None
        self.mask = None
        self.prediction = None
        self.tracklet_mask_dict = None
        self.mask_color_counter = 0
        self.current_object_list_cutie = []
        self.last_object_number_cutie = 0
        self.awaiting_mask_tracklet_ids = []
        self.init_delay_counter = 0
        self.num_objects = 0
        self.mask_prediction_prev_frame = None
        self.mask_color_dict = {}

        # Initialize SAM2.1 via HuggingFace
        print(f"Loading SAM2.1 model: {sam2_model_id}")
        token = hf_token if hf_token else None

        # Load SAM2 for image segmentation
        try:
            from transformers import Sam2Processor, Sam2Model

            self.sam2_processor = Sam2Processor.from_pretrained(
                sam2_model_id,
                token=token,
            )
            self.sam2_model = Sam2Model.from_pretrained(
                sam2_model_id,
                token=token,
            )
            self.sam2_model.to(device).eval()
            self._sam2_use_hf = True
            print("SAM2.1 loaded successfully via HuggingFace")
        except Exception as e:
            print(f"Warning: Failed to load SAM2 from HuggingFace: {e}")
            print("Trying alternative SAM2 loading method...")
            self._sam2_use_hf = False
            self.sam2_model = None
            self.sam2_processor = None

        # Initialize Cutie
        self._init_cutie(cutie_weights_path)

    def _init_cutie(self, cutie_weights_path: Optional[str] = None):
        """Initialize Cutie model for temporal mask propagation."""
        try:
            # Try to import Cutie
            from omegaconf import OmegaConf, open_dict
            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra

            # Find Cutie installation
            cutie_root = self._find_cutie_root()
            if cutie_root is None:
                print("Warning: Cutie not found. Mask propagation will be disabled.")
                self.cutie = None
                self.processor = None
                return

            sys.path.insert(0, str(cutie_root))

            from cutie.model.cutie import CUTIE
            from cutie.inference.inference_core import InferenceCore
            from cutie.inference.utils.args_utils import get_dataset_cfg

            # Determine weights path
            if cutie_weights_path is None:
                cutie_weights_path = cutie_root / "weights" / "cutie-base-mega.pth"
            else:
                cutie_weights_path = Path(cutie_weights_path)

            if not cutie_weights_path.exists():
                print(f"Warning: Cutie weights not found at {cutie_weights_path}")
                print("Please download cutie-base-mega.pth from https://github.com/hkchengrex/Cutie")
                self.cutie = None
                self.processor = None
                return

            # Initialize Hydra config
            config_path = cutie_root / "cutie" / "config"

            # Clear any existing Hydra instance
            GlobalHydra.instance().clear()

            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=True):
                    initialize_config_dir(version_base='1.3.2', config_dir=str(config_path))
                    cfg = compose(config_name="eval_config")

                    with open_dict(cfg):
                        cfg['weights'] = str(cutie_weights_path)

                    # This function modifies cfg with additional values
                    _ = get_dataset_cfg(cfg)

                    # Load model
                    self.cutie = CUTIE(cfg).to(self.device).eval()
                    model_weights = torch.load(cutie_weights_path, map_location=self.device)
                    self.cutie.load_weights(model_weights)

                    # Initialize inference core
                    torch.cuda.empty_cache()
                    self.processor = InferenceCore(self.cutie, cfg=cfg)

            print(f"Cutie loaded successfully from {cutie_weights_path}")

        except Exception as e:
            print(f"Warning: Failed to initialize Cutie: {e}")
            print("Mask propagation will be disabled.")
            self.cutie = None
            self.processor = None

    def _find_cutie_root(self) -> Optional[Path]:
        """Find the Cutie installation directory."""
        # Check common locations
        possible_paths = [
            Path(__file__).parent / "Cutie",
            Path(__file__).parent.parent / "Cutie",
            Path.home() / "Cutie",
            Path("/content/Cutie"),  # Colab
        ]

        for path in possible_paths:
            if path.exists() and (path / "cutie").exists():
                return path

        return None

    def _sam2_predict_boxes(self, image: np.ndarray, boxes_xyxy: List[List[float]]) -> np.ndarray:
        """Generate masks for multiple bounding boxes using SAM2.1 with batching.

        Parameters
        ----------
        image : np.ndarray
            RGB image as numpy array (H, W, 3)
        boxes_xyxy : List[List[float]]
            List of [x1, y1, x2, y2] bounding boxes

        Returns
        -------
        masks : np.ndarray
            Binary masks as numpy array (N, H, W)
        """
        H, W = image.shape[:2]

        if len(boxes_xyxy) == 0:
            return np.zeros((0, H, W), dtype=np.uint8)

        if self.sam2_model is None:
            # Fallback: use bounding box as mask
            return self._bbox_to_mask_fallback(image, boxes_xyxy)

        try:
            # BATCH ALL BOXES IN ONE CALL for performance
            input_boxes = [[[box[0], box[1], box[2], box[3]]] for box in boxes_xyxy]

            inputs = self.sam2_processor(
                images=[image] * len(boxes_xyxy),
                input_boxes=input_boxes,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.sam2_model(**inputs, multimask_output=False)

            pred_masks = outputs.pred_masks  # (N, 1, 1, H', W')
            pred_masks = F.interpolate(
                pred_masks.squeeze(2).float(),  # (N, 1, H', W')
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
            masks = (pred_masks.squeeze(1) > 0.5).cpu().numpy().astype(np.uint8)
            return masks

        except Exception as e:
            print(f"Batch SAM2 failed: {e}, falling back to sequential")
            return self._sam2_predict_boxes_sequential(image, boxes_xyxy)

    def _sam2_predict_boxes_sequential(self, image: np.ndarray, boxes_xyxy: List[List[float]]) -> np.ndarray:
        """Fallback: Generate masks sequentially when batching fails.

        Parameters
        ----------
        image : np.ndarray
            RGB image as numpy array (H, W, 3)
        boxes_xyxy : List[List[float]]
            List of [x1, y1, x2, y2] bounding boxes

        Returns
        -------
        masks : np.ndarray
            Binary masks as numpy array (N, H, W)
        """
        H, W = image.shape[:2]

        try:
            all_masks = []

            for box in boxes_xyxy:
                # Format single box for SAM2: [[[x1, y1, x2, y2]]]
                input_boxes = [[box]]

                # Process with SAM2.1
                inputs = self.sam2_processor(
                    images=image,
                    input_boxes=input_boxes,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.sam2_model(**inputs, multimask_output=False)

                # Get the predicted mask
                pred_masks = outputs.pred_masks  # Shape: (1, 1, 1, H', W')

                # Resize to original image size
                pred_masks = F.interpolate(
                    pred_masks.squeeze(0).float(),  # (1, 1, H', W')
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                )

                # Threshold and convert to binary
                mask = (pred_masks.squeeze() > 0.5).cpu().numpy().astype(np.uint8)
                all_masks.append(mask)

            return np.stack(all_masks, axis=0)

        except Exception as e:
            print(f"Warning: SAM2.1 sequential prediction failed: {e}")
            print("Using bounding box fallback for masks")
            return self._bbox_to_mask_fallback(image, boxes_xyxy)

    def _bbox_to_mask_fallback(self, image: np.ndarray, boxes_xyxy: List[List[float]]) -> np.ndarray:
        """Fallback: create masks from bounding boxes directly.

        Parameters
        ----------
        image : np.ndarray
            RGB image
        boxes_xyxy : List[List[float]]
            Bounding boxes

        Returns
        -------
        masks : np.ndarray
            Binary masks (N, H, W)
        """
        H, W = image.shape[:2]
        masks = []

        for box in boxes_xyxy:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)

            mask = np.zeros((H, W), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1
            masks.append(mask)

        return np.stack(masks, axis=0) if masks else np.zeros((0, H, W), dtype=np.uint8)

    def get_updated_masks(
        self,
        img_info: dict,
        img_info_prev: dict,
        frame_id: int,
        online_tlwhs: List,
        online_ids: List[int],
        new_tracks: List,
        removed_tracks_ids: List[int]
    ) -> Tuple[Optional[np.ndarray], Dict, Optional[Dict], Optional[np.ndarray]]:
        """Propagate existing masks and create/remove masks as needed.

        Parameters
        ----------
        img_info : dict
            Current frame information dict with 'raw_img' key
        img_info_prev : dict
            Previous frame information dict with 'raw_img' key
        frame_id : int
            Current frame number (starting from 1)
        online_tlwhs : List
            Updated tracklet positions [top, left, width, height]
        online_ids : List[int]
            Updated tracklet IDs
        new_tracks : List
            Newly created tracklets
        removed_tracks_ids : List[int]
            IDs of removed tracklets

        Returns
        -------
        prediction : np.ndarray or None
            Propagated mask (H, W) with unique values per object
        tracklet_mask_dict : dict
            Mapping of track_id -> mask color value
        mask_avg_prob_dict : dict or None
            Mapping of mask_id -> confidence (0-1)
        prediction_colors_preserved : np.ndarray or None
            Mask with preserved colors for visualization
        """
        if self.processor is None:
            # Cutie not available, return empty results
            return None, {}, None, None

        # Convert numpy arrays to torch tensors for Cutie
        frame_torch = self._image_to_torch(img_info['raw_img'])
        frame_torch_prev = self._image_to_torch(img_info_prev['raw_img'])

        prediction = self.prediction

        # First frame with masks: initialize all tracklets
        if frame_id == self.SAM_START_FRAME + 1 + self.init_delay_counter and online_tlwhs is not None:
            prediction = self.initialize_first_masks(
                frame_torch, frame_torch_prev, img_info_prev, online_tlwhs, online_ids
            )

        # Subsequent frames: update masks
        elif frame_id > self.SAM_START_FRAME + 1 + self.init_delay_counter:
            self.add_new_masks(frame_torch_prev, img_info_prev, online_tlwhs, online_ids, new_tracks)
            self.remove_masks(removed_tracks_ids)

            # Continue propagation
            with torch.no_grad():
                prediction = self.processor.step(frame_torch)

        self.prediction = prediction

        mask_avg_prob_dict = None
        prediction_colors_preserved = None

        if prediction is not None:
            prediction, mask_avg_prob_dict, prediction_colors_preserved = self.post_process_mask(prediction)

        tracklet_mask_dict = self.tracklet_mask_dict.copy() if self.tracklet_mask_dict else {}

        return prediction, tracklet_mask_dict, mask_avg_prob_dict, prediction_colors_preserved

    def initialize_first_masks(
        self,
        frame_torch: torch.Tensor,
        frame_torch_prev: torch.Tensor,
        img_info_prev: dict,
        online_tlwhs: List,
        online_ids: List[int]
    ) -> Optional[torch.Tensor]:
        """Create initial masks for all tracklets on first frame.

        Parameters
        ----------
        frame_torch : torch.Tensor
            Current frame tensor
        frame_torch_prev : torch.Tensor
            Previous frame tensor
        img_info_prev : dict
            Previous frame info with 'raw_img'
        online_tlwhs : List
            Tracklet positions [top, left, width, height]
        online_ids : List[int]
            Tracklet IDs

        Returns
        -------
        prediction : torch.Tensor or None
            Initial mask prediction
        """
        image_boxes_list = []
        new_tracks_id = []

        for i, ot in enumerate(online_tlwhs):
            # Avoid creating masks for overlapped subjects
            track_BBs_with_lower_bottom = get_tracklets_with_lower_bottom(ot, online_tlwhs)
            overlap = get_overlap_with_lower_bottom_tracklets(ot, track_BBs_with_lower_bottom)

            if overlap >= self.bbox_overlap_threshold:
                self.awaiting_mask_tracklet_ids.append(online_ids[i])
                continue

            # Convert tlwh to xyxy
            image_boxes_list.append([ot[0], ot[1], ot[0] + ot[2], ot[1] + ot[3]])
            new_tracks_id.append(online_ids[i])

        if len(image_boxes_list) == 0:
            self.init_delay_counter += 1
            return None

        # Generate masks with SAM2.1
        masks = self._sam2_predict_boxes(img_info_prev['raw_img'], image_boxes_list)

        # Convert individual masks to combined mask with unique IDs
        H, W = masks.shape[1], masks.shape[2]
        mask = np.zeros((H, W), dtype=np.int32)

        for mi in range(len(masks)):
            current_mask = masks[mi].astype(np.int32)
            current_mask[current_mask > 0] = mi + 1

            non_occupied = (mask == 0).astype(np.int32)
            mask += (current_mask * non_occupied)

        self.num_objects = len(masks)

        # Update object tracking lists
        self.current_object_list_cutie = list(range(1, self.num_objects + 1))
        self.last_object_number_cutie = max(self.current_object_list_cutie, default=0)

        # Create tracklet -> mask mapping
        self.tracklet_mask_dict = dict(zip(new_tracks_id, range(1, self.num_objects + 1)))

        # Create color mapping for visualization
        self.mask_color_dict = dict(zip(range(1, self.num_objects + 1), range(1, self.num_objects + 1)))
        self.mask_color_counter = max(list(self.mask_color_dict.values()), default=0)

        # Initialize Cutie with the masks
        mask_torch = self._index_numpy_to_one_hot_torch(mask, self.num_objects + 1).to(self.device)
        with torch.no_grad():
            _ = self.processor.step(frame_torch_prev, mask_torch[1:], idx_mask=False)
            prediction = self.processor.step(frame_torch)

        return prediction

    def add_new_masks(
        self,
        frame_torch_prev: torch.Tensor,
        img_info_prev: dict,
        online_tlwhs: List,
        online_ids: List[int],
        new_tracks: List
    ):
        """Create masks for newly initialized tracklets.

        Parameters
        ----------
        frame_torch_prev : torch.Tensor
            Previous frame tensor
        img_info_prev : dict
            Previous frame info with 'raw_img'
        online_tlwhs : List
            Current tracklet positions
        online_ids : List[int]
            Current tracklet IDs
        new_tracks : List
            List of newly created STrack objects
        """
        if len(new_tracks) == 0 and len(self.awaiting_mask_tracklet_ids) == 0:
            return

        image_boxes_list = []
        new_tracks_id = []

        # Process awaiting tracklets first
        for amti in list(self.awaiting_mask_tracklet_ids):
            if amti not in online_ids:
                continue

            amt_index = online_ids.index(amti)
            amt_tlwh = online_tlwhs[amt_index]

            track_BBs_with_lower_bottom = get_tracklets_with_lower_bottom(amt_tlwh, online_tlwhs)
            overlap = get_overlap_with_lower_bottom_tracklets(amt_tlwh, track_BBs_with_lower_bottom)

            if overlap < self.bbox_overlap_threshold:
                image_boxes_list.append([amt_tlwh[0], amt_tlwh[1],
                                        amt_tlwh[0] + amt_tlwh[2], amt_tlwh[1] + amt_tlwh[3]])
                new_tracks_id.append(amti)

        # Remove processed tracklets from awaiting list
        for nti in new_tracks_id:
            if nti in self.awaiting_mask_tracklet_ids:
                self.awaiting_mask_tracklet_ids.remove(nti)

        # Process new tracklets
        for nt in new_tracks:
            track_BBs_with_lower_bottom = get_tracklets_with_lower_bottom(nt.last_det_tlwh, online_tlwhs)
            overlap = get_overlap_with_lower_bottom_tracklets(nt.last_det_tlwh, track_BBs_with_lower_bottom)

            if overlap >= self.bbox_overlap_threshold:
                self.awaiting_mask_tracklet_ids.append(nt.track_id)
                continue

            tlwh = nt.last_det_tlwh
            image_boxes_list.append([tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]])
            new_tracks_id.append(nt.track_id)

        if len(image_boxes_list) == 0:
            return

        # Generate masks with SAM2.1
        masks = self._sam2_predict_boxes(img_info_prev['raw_img'], image_boxes_list)

        # Create combined mask for new objects
        H, W = masks.shape[1], masks.shape[2]
        mask_extra = np.zeros((H, W), dtype=np.int32)
        max_mask_number = max(self.tracklet_mask_dict.values(), default=0)

        new_masks_numbers = []
        new_object_numbers = []

        for mi in range(len(masks)):
            current_mask = masks[mi].astype(np.int32)
            next_mask_number = max_mask_number + mi + 1
            current_mask[current_mask > 0] = next_mask_number
            new_masks_numbers.append(next_mask_number)

            self.last_object_number_cutie += 1
            new_object_numbers.append(self.last_object_number_cutie)

            non_occupied = (mask_extra == 0).astype(np.int32)
            mask_extra += (current_mask * non_occupied)

        # Update previous frame mask with new masks
        if self.mask_prediction_prev_frame is not None:
            self.mask_prediction_prev_frame[mask_extra > 0] = mask_extra[mask_extra > 0]
        else:
            self.mask_prediction_prev_frame = mask_extra

        self.num_objects += len(new_tracks_id)

        # Convert to torch and add to Cutie
        mask_prev_extended_torch = self._index_numpy_to_one_hot_torch(
            self.mask_prediction_prev_frame, self.num_objects + 1
        ).to(self.device)

        self.current_object_list_cutie.extend(new_object_numbers)

        # Incorporate new masks into Cutie memory
        with torch.no_grad():
            _ = self.processor.step(
                frame_torch_prev, mask_prev_extended_torch[1:],
                objects=self.current_object_list_cutie, idx_mask=False
            )

        # Update dictionaries
        self.mask_color_counter = update_tracklet_mask_dict_after_mask_addition(
            self.tracklet_mask_dict, self.mask_color_dict,
            new_tracks_id, new_masks_numbers, self.mask_color_counter
        )

    def remove_masks(self, removed_tracks_ids: List[int]):
        """Remove masks for removed tracklets.

        Parameters
        ----------
        removed_tracks_ids : List[int]
            IDs of removed tracklets
        """
        if len(removed_tracks_ids) == 0 or self.tracklet_mask_dict is None:
            return

        mask_ids_to_be_removed = [
            self.tracklet_mask_dict[i]
            for i in self.tracklet_mask_dict.keys()
            if i in removed_tracks_ids
        ]

        if len(mask_ids_to_be_removed) == 0:
            return

        # Use InferenceCore's delete_objects which handles both object_manager and memory
        self.processor.delete_objects(mask_ids_to_be_removed)

        # Update internal state
        self.current_object_list_cutie = self.processor.object_manager.all_obj_ids
        self.num_objects = len(self.current_object_list_cutie)

        self.mask_color_counter = update_tracklet_mask_dict_after_mask_removal(
            self.tracklet_mask_dict, self.mask_color_dict, mask_ids_to_be_removed
        )

    def post_process_mask(
        self, prediction: torch.Tensor
    ) -> Tuple[np.ndarray, Dict[int, float], np.ndarray]:
        """Post-process Cutie output to numpy mask.

        Parameters
        ----------
        prediction : torch.Tensor
            Raw prediction from Cutie

        Returns
        -------
        prediction : np.ndarray
            Processed mask (H, W)
        mask_avg_prob_dict : dict
            Mapping of mask_id -> average confidence
        prediction_colors_preserved : np.ndarray
            Mask with preserved colors
        """
        mask_avg_prob_dict = self._get_mask_avg_prob(prediction)

        # Convert to numpy
        prediction_np = self._torch_prob_to_numpy_mask(prediction)

        self.mask_prediction_prev_frame = prediction_np.copy()

        # Preserve colors for visualization
        prediction_colors_preserved = self._adjust_mask_colors(prediction_np)

        return prediction_np, mask_avg_prob_dict, prediction_colors_preserved

    def _get_mask_avg_prob(self, prediction: torch.Tensor) -> Dict[int, float]:
        """Calculate average probability for each mask."""
        mask_avg_prob_dict = {}
        mask_maxes = torch.max(prediction, dim=0).indices

        if self.tracklet_mask_dict is None:
            return mask_avg_prob_dict

        for v in self.tracklet_mask_dict.values():
            if v < prediction.shape[0]:
                avg_score = (prediction[v][mask_maxes == v]).mean().item()
                if avg_score is not None and not np.isnan(avg_score):
                    mask_avg_prob_dict[v] = avg_score

        return mask_avg_prob_dict

    def _adjust_mask_colors(self, prediction: np.ndarray) -> np.ndarray:
        """Preserve colors for masks after ID shifts."""
        prediction_colors_preserved = prediction.copy()
        keys_descending = sorted(list(self.mask_color_dict.keys()), reverse=True)

        for k in keys_descending:
            prediction_colors_preserved[prediction_colors_preserved == k] = self.mask_color_dict[k]

        return prediction_colors_preserved

    def _image_to_torch(self, frame: np.ndarray) -> torch.Tensor:
        """Convert numpy image to torch tensor."""
        frame = frame.transpose(2, 0, 1)
        frame = torch.from_numpy(frame).float().to(self.device, non_blocking=True) / 255
        return frame

    def _torch_prob_to_numpy_mask(self, prob: torch.Tensor) -> np.ndarray:
        """Convert torch probability to numpy mask."""
        mask = torch.max(prob, dim=0).indices
        mask = mask.cpu().numpy().astype(np.uint8)
        return mask

    def _index_numpy_to_one_hot_torch(self, mask: np.ndarray, num_classes: int) -> torch.Tensor:
        """Convert numpy index mask to one-hot torch tensor."""
        mask = torch.from_numpy(mask).long()
        return F.one_hot(mask, num_classes=num_classes).permute(2, 0, 1).float()


# Helper functions for overlap calculation

def get_tracklets_with_lower_bottom(new_tracklet_tlwh, online_tlwhs):
    """Find tracklets with bottom coordinate lower than the given tracklet."""
    nt_y = new_tracklet_tlwh[1]
    nt_h = new_tracklet_tlwh[3]

    track_BBs_with_lower_bottom = []
    nt_bottom = nt_y + nt_h

    for ot in online_tlwhs:
        if ot[1] + ot[3] > nt_bottom:
            track_BBs_with_lower_bottom.append(ot)

    return track_BBs_with_lower_bottom


def get_overlap_with_lower_bottom_tracklets(new_tracklet_tlwh, track_BBs_with_lower_bottom):
    """Calculate overlap with lower-positioned tracklets."""
    if OVERLAP_MEASURE_VARIANT == 1:
        return get_overlap_variant_1(new_tracklet_tlwh, track_BBs_with_lower_bottom)
    elif OVERLAP_MEASURE_VARIANT == 2:
        return get_overlap_variant_2(new_tracklet_tlwh, track_BBs_with_lower_bottom)
    return 0


def get_overlap_variant_1(new_tracklet_tlwh, track_BBs_with_lower_bottom):
    """Calculate max overlap ratio using intersection."""
    nt_x = new_tracklet_tlwh[0]
    nt_y = new_tracklet_tlwh[1]
    nt_w = new_tracklet_tlwh[2]
    nt_h = new_tracklet_tlwh[3]

    max_overlap_part = 0

    for lb in track_BBs_with_lower_bottom:
        x_dist = min(nt_x + nt_w, lb[0] + lb[2]) - max(nt_x, lb[0])
        y_dist = min(nt_y + nt_h, lb[1] + lb[3]) - max(nt_y, lb[1])

        if x_dist < 0 or y_dist < 0:
            overlap_area = 0
        else:
            overlap_area = x_dist * y_dist

        if nt_w * nt_h > 0:
            overlap_part = overlap_area / (nt_w * nt_h)
            if max_overlap_part < overlap_part:
                max_overlap_part = overlap_part

                if max_overlap_part == 1:
                    break

    return max_overlap_part


def get_overlap_variant_2(new_tracklet_tlwh, track_BBs_with_lower_bottom):
    """Calculate overlap using grid sampling."""
    nt_x = int(new_tracklet_tlwh[0])
    nt_y = int(new_tracklet_tlwh[1])
    nt_w = int(new_tracklet_tlwh[2])
    nt_h = int(new_tracklet_tlwh[3])

    point_overlap_counter = 0

    y_range = list(range(nt_y, nt_y + nt_h, OVERLAP_VARIANT_2_GRID_STEP))
    x_range = list(range(nt_x, nt_x + nt_w, OVERLAP_VARIANT_2_GRID_STEP))

    if len(y_range) == 0 or len(x_range) == 0:
        return 0

    for grid_row in y_range:
        for grid_col in x_range:
            for lb in track_BBs_with_lower_bottom:
                if (lb[0] <= grid_col <= lb[0] + lb[2] and
                    lb[1] <= grid_row <= lb[1] + lb[3]):
                    point_overlap_counter += 1
                    break

    overlap_part = point_overlap_counter / (len(y_range) * len(x_range))
    return overlap_part


def update_tracklet_mask_dict_after_mask_addition(
    tracklet_mask_dict, mask_color_dict, added_tracklet_ids, added_mask_ids, mask_color_counter
):
    """Update dictionaries after adding new masks."""
    # Update tracklet -> mask mapping
    for k, v in zip(added_tracklet_ids, added_mask_ids):
        tracklet_mask_dict[k] = v

    # Update color mapping
    for mi in added_mask_ids:
        mask_color_counter += 1
        mask_color_dict[mi] = mask_color_counter

    return mask_color_counter


def update_tracklet_mask_dict_after_mask_removal(
    tracklet_mask_dict, mask_color_dict, removed_mask_ids
):
    """Update dictionaries after removing masks."""
    # Update tracklet_mask_dict
    entries_to_be_removed = []
    decrement_mask_id_dict = {}

    for k in tracklet_mask_dict.keys():
        if tracklet_mask_dict[k] in removed_mask_ids:
            entries_to_be_removed.append(k)
        else:
            for rmi in removed_mask_ids:
                if tracklet_mask_dict[k] > rmi:
                    if k not in decrement_mask_id_dict:
                        decrement_mask_id_dict[k] = 1
                    else:
                        decrement_mask_id_dict[k] += 1

    for entry in entries_to_be_removed:
        del tracklet_mask_dict[entry]

    for k in decrement_mask_id_dict:
        tracklet_mask_dict[k] -= decrement_mask_id_dict[k]

    # Update mask_color_dict
    mask_color_counter = max(list(mask_color_dict.values()), default=0)

    entries_to_be_removed = []
    decrement_mask_id_dict = {}

    for k in mask_color_dict.keys():
        if k in removed_mask_ids:
            entries_to_be_removed.append(k)
        else:
            for rmi in removed_mask_ids:
                if k > rmi:
                    if k not in decrement_mask_id_dict:
                        decrement_mask_id_dict[k] = 1
                    else:
                        decrement_mask_id_dict[k] += 1

    for entry in entries_to_be_removed:
        del mask_color_dict[entry]

    mask_color_keys = list(decrement_mask_id_dict.keys())
    for mc in mask_color_keys:
        new_key = mc - decrement_mask_id_dict[mc]
        mask_color_dict[new_key] = mask_color_dict[mc]
        del mask_color_dict[mc]

    return mask_color_counter
