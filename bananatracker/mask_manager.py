"""Mask propagation manager using SAM2.1 for temporal mask tracking.

This module provides mask generation from bounding boxes and temporal propagation
using SAM2.1's unified video object segmentation pipeline.

Usage:
    from bananatracker.mask_manager import MaskManager

    mask_manager = MaskManager(config)
    masks, tracklet_mask_dict, confidence, colors = mask_manager.get_updated_masks(...)
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .config import BananaTrackerConfig


# Overlap threshold for skipping mask creation on heavily occluded objects
MASK_CREATION_BBOX_OVERLAP_THRESHOLD = 0.6


class MaskManager:
    """Manages mask creation and temporal propagation using SAM2.1."""

    def __init__(self, config: BananaTrackerConfig):
        """Initialize the mask manager with SAM2.1.

        Parameters
        ----------
        config : BananaTrackerConfig
            Configuration object with SAM2.1 settings.
        """
        self.config = config
        self.device = config.device

        # SAM2.1 predictor (lazy initialization)
        self.predictor = None
        self.first_frame_loaded = False

        # Object tracking state
        self.tracklet_to_sam2_obj = {}  # track_id -> SAM2.1 obj_id
        self.sam2_obj_counter = 0
        self.removed_obj_ids = set()  # Soft-removed objects (filtered in output)
        self.frame_counter = 0

        # Mask state
        self.tracklet_mask_dict = {}  # track_id -> mask_id (1-indexed)
        self.mask_color_dict = {}  # mask_id -> color_id (for visualization)
        self.mask_color_counter = 0
        self.num_objects = 0
        self.mask_prediction_prev_frame = None

        # Deferred mask creation for overlapping objects
        self.awaiting_mask_tracklet_ids = []
        self.init_delay_counter = 0
        self.mask_start_frame = config.mask_start_frame

        # Initialize SAM2.1
        if config.sam2_enabled:
            self._init_sam2()

    def _init_sam2(self):
        """Initialize the SAM2.1 camera predictor."""
        # Determine SAM2.1 repo path
        sam2_repo_path = self.config.sam2_repo_path
        if not sam2_repo_path:
            # Try common locations
            possible_paths = [
                Path(__file__).parent.parent.parent / "segment-anything-2-real-time",
                Path.home() / "segment-anything-2-real-time",
                Path("/Users/harry/final/segment-anything-2-real-time"),
            ]
            for p in possible_paths:
                if p.exists():
                    sam2_repo_path = str(p)
                    break

        if not sam2_repo_path or not Path(sam2_repo_path).exists():
            raise RuntimeError(
                f"SAM2.1 repo not found. Please set sam2_repo_path in config. "
                f"Tried: {possible_paths}"
            )

        # Add SAM2.1 to path
        if sam2_repo_path not in sys.path:
            sys.path.insert(0, sam2_repo_path)

        # Pre-import all SAM2 modules that Hydra needs to instantiate.
        # This ensures the modules are in sys.modules before Hydra's instantiate()
        # tries to resolve _target_ classes from the YAML config.
        import sam2.modeling.sam2_base  # noqa: F401
        import sam2.modeling.backbones.image_encoder  # noqa: F401
        import sam2.modeling.backbones.hieradet  # noqa: F401
        import sam2.modeling.position_encoding  # noqa: F401
        import sam2.modeling.memory_attention  # noqa: F401
        import sam2.modeling.sam.transformer  # noqa: F401
        import sam2.modeling.memory_encoder  # noqa: F401
        import sam2.sam2_camera_predictor  # noqa: F401

        # Build checkpoint and config paths
        checkpoint_path = self.config.sam2_checkpoint
        config_path = self.config.sam2_config

        # If paths are relative, make them absolute to SAM2.1 repo
        if not Path(checkpoint_path).is_absolute():
            checkpoint_path = str(Path(sam2_repo_path) / checkpoint_path)
        if not Path(config_path).is_absolute():
            config_path = str(Path(sam2_repo_path) / config_path)

        # Verify checkpoint exists
        if not Path(checkpoint_path).exists():
            raise RuntimeError(
                f"SAM2.1 checkpoint not found at {checkpoint_path}. "
                "Please download checkpoints using: cd segment-anything-2-real-time/checkpoints && bash download_ckpts.sh"
            )

        # Import and build predictor
        from sam2.build_sam import build_sam2_camera_predictor

        # Use relative config name for hydra
        config_name = Path(config_path).stem

        self.predictor = build_sam2_camera_predictor(
            config_file=config_name,
            ckpt_path=checkpoint_path,
            device=self.device,
        )

        print(f"SAM2.1 initialized with checkpoint: {checkpoint_path}")

    def reset(self):
        """Reset the mask manager state for a new video."""
        self.first_frame_loaded = False
        self.tracklet_to_sam2_obj = {}
        self.sam2_obj_counter = 0
        self.removed_obj_ids = set()
        self.frame_counter = 0
        self.tracklet_mask_dict = {}
        self.mask_color_dict = {}
        self.mask_color_counter = 0
        self.num_objects = 0
        self.mask_prediction_prev_frame = None
        self.awaiting_mask_tracklet_ids = []
        self.init_delay_counter = 0

        # Re-initialize predictor state
        if self.predictor is not None:
            self.predictor.condition_state = {}
            self.predictor.frame_idx = 0

    def get_updated_masks(
        self,
        frame: np.ndarray,
        frame_prev: Optional[np.ndarray],
        frame_id: int,
        online_tlwhs: List,
        online_ids: List,
        new_tracks: List,
        removed_tracks_ids: List,
    ) -> Tuple[Optional[np.ndarray], Dict[int, int], Dict[int, float], Optional[np.ndarray]]:
        """Update masks based on tracker state.

        This method handles mask initialization, propagation, addition, and removal
        based on the current frame and tracker output.

        Parameters
        ----------
        frame : np.ndarray
            Current frame (BGR).
        frame_prev : np.ndarray, optional
            Previous frame (BGR). Required for initialization.
        frame_id : int
            Current frame number (1-indexed).
        online_tlwhs : List
            List of current tracklet positions as [top, left, width, height].
        online_ids : List
            List of current tracklet IDs.
        new_tracks : List[STrack]
            Newly created tracklets.
        removed_tracks_ids : List[int]
            IDs of removed tracklets.

        Returns
        -------
        prediction : np.ndarray or None
            Mask prediction of shape (H, W) with integer IDs (0=background).
        tracklet_mask_dict : Dict[int, int]
            Mapping from track_id to mask_id.
        mask_avg_prob_dict : Dict[int, float]
            Mapping from mask_id to average confidence.
        prediction_colors : np.ndarray or None
            Color-preserved mask for visualization.
        """
        if not self.config.sam2_enabled or self.predictor is None:
            return None, {}, {}, None

        prediction = None

        # Determine when to initialize based on mask_start_frame
        init_frame = self.mask_start_frame + 1 + self.init_delay_counter

        if frame_id == init_frame and frame_prev is not None:
            # Initialize masks on first applicable frame
            prediction = self._initialize_first_masks(frame, frame_prev, online_tlwhs, online_ids)

        elif frame_id > init_frame and self.first_frame_loaded:
            # Add new masks for new tracklets
            if frame_prev is not None:
                self._add_new_masks(frame_prev, online_tlwhs, online_ids, new_tracks)

            # Remove masks for removed tracklets
            self._remove_masks(removed_tracks_ids)

            # Propagate masks to current frame
            prediction = self._propagate_masks(frame)

        # Post-process prediction
        mask_avg_prob_dict = {}
        prediction_colors = None

        if prediction is not None:
            prediction, mask_avg_prob_dict, prediction_colors = self._post_process_mask(prediction)

        return prediction, self.tracklet_mask_dict.copy(), mask_avg_prob_dict, prediction_colors

    def _initialize_first_masks(
        self,
        frame: np.ndarray,
        frame_prev: np.ndarray,
        online_tlwhs: List,
        online_ids: List,
    ) -> Optional[torch.Tensor]:
        """Initialize masks for all current tracklets.

        Parameters
        ----------
        frame : np.ndarray
            Current frame (BGR).
        frame_prev : np.ndarray
            Previous frame (BGR) - used for mask creation.
        online_tlwhs : List
            List of tracklet positions.
        online_ids : List
            List of tracklet IDs.

        Returns
        -------
        prediction : torch.Tensor or None
            Mask prediction tensor.
        """
        if online_tlwhs is None or len(online_tlwhs) == 0:
            self.init_delay_counter += 1
            return None

        # Load first frame into SAM2.1
        self.predictor.load_first_frame(frame_prev)
        self.first_frame_loaded = True

        # Filter overlapping tracklets
        image_boxes_list = []
        new_tracks_id = []

        for i, tlwh in enumerate(online_tlwhs):
            # Check for heavy overlap with lower objects
            track_BBs_with_lower_bottom = self._get_tracklets_with_lower_bottom(tlwh, online_tlwhs)
            overlap = self._get_overlap_with_lower_bottom_tracklets(tlwh, track_BBs_with_lower_bottom)

            if overlap >= self.config.mask_overlap_threshold:
                self.awaiting_mask_tracklet_ids.append(online_ids[i])
                continue

            # Convert tlwh to xyxy
            x1, y1, w, h = tlwh
            image_boxes_list.append([x1, y1, x1 + w, y1 + h])
            new_tracks_id.append(online_ids[i])

        if len(image_boxes_list) == 0:
            self.init_delay_counter += 1
            return None

        # Add each object to SAM2.1
        all_masks = []
        H, W = frame_prev.shape[:2]

        for bbox, track_id in zip(image_boxes_list, new_tracks_id):
            self.sam2_obj_counter += 1
            sam2_obj_id = self.sam2_obj_counter

            # Convert bbox to SAM2.1 format: [[x1, y1], [x2, y2]]
            bbox_sam2 = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)

            # Add prompt to SAM2.1
            _, obj_ids, mask_logits = self.predictor.add_new_prompt(
                frame_idx=0,
                obj_id=sam2_obj_id,
                bbox=bbox_sam2,
            )

            self.tracklet_to_sam2_obj[track_id] = sam2_obj_id
            all_masks.append((sam2_obj_id, mask_logits))

        # Build tracklet_mask_dict and mask_color_dict
        self.num_objects = len(new_tracks_id)
        self.tracklet_mask_dict = dict(zip(new_tracks_id, range(1, self.num_objects + 1)))
        self.mask_color_dict = dict(zip(range(1, self.num_objects + 1), range(1, self.num_objects + 1)))
        self.mask_color_counter = self.num_objects

        # Propagate to current frame
        self.frame_counter = 0
        prediction = self._propagate_masks(frame)

        return prediction

    def _add_new_masks(
        self,
        frame_prev: np.ndarray,
        online_tlwhs: List,
        online_ids: List,
        new_tracks: List,
    ):
        """Add masks for newly created tracklets.

        Parameters
        ----------
        frame_prev : np.ndarray
            Previous frame (BGR).
        online_tlwhs : List
            List of current tracklet positions.
        online_ids : List
            List of current tracklet IDs.
        new_tracks : List[STrack]
            Newly created tracklets.
        """
        if len(new_tracks) == 0 and len(self.awaiting_mask_tracklet_ids) == 0:
            return

        image_boxes_list = []
        new_tracks_id = []

        # Try to create masks for previously deferred tracklets
        tracklets_to_remove_from_awaiting = []
        for amti in self.awaiting_mask_tracklet_ids:
            if amti not in online_ids:
                continue

            amt_index = online_ids.index(amti)
            amt_tlwh = online_tlwhs[amt_index]

            track_BBs_with_lower_bottom = self._get_tracklets_with_lower_bottom(amt_tlwh, online_tlwhs)
            overlap = self._get_overlap_with_lower_bottom_tracklets(amt_tlwh, track_BBs_with_lower_bottom)

            if overlap < self.config.mask_overlap_threshold:
                x1, y1, w, h = amt_tlwh
                image_boxes_list.append([x1, y1, x1 + w, y1 + h])
                new_tracks_id.append(amti)
                tracklets_to_remove_from_awaiting.append(amti)

        for nti in tracklets_to_remove_from_awaiting:
            self.awaiting_mask_tracklet_ids.remove(nti)

        # Check new tracks for overlap
        for nt in new_tracks:
            tlwh = nt.tlwh if hasattr(nt, 'tlwh') else nt.last_det_tlwh

            track_BBs_with_lower_bottom = self._get_tracklets_with_lower_bottom(tlwh, online_tlwhs)
            overlap = self._get_overlap_with_lower_bottom_tracklets(tlwh, track_BBs_with_lower_bottom)

            if overlap >= self.config.mask_overlap_threshold:
                self.awaiting_mask_tracklet_ids.append(nt.track_id)
                continue

            x1, y1, w, h = tlwh
            image_boxes_list.append([x1, y1, x1 + w, y1 + h])
            new_tracks_id.append(nt.track_id)

        if len(image_boxes_list) == 0:
            return

        # Add conditioning frame if tracking has started
        if self.predictor.condition_state.get("tracking_has_started", False):
            # Reset tracking state to add new objects
            self.predictor.condition_state["tracking_has_started"] = False

        # Add each new object
        max_mask_number = max(self.tracklet_mask_dict.values(), default=0)
        new_masks_numbers = []

        for i, (bbox, track_id) in enumerate(zip(image_boxes_list, new_tracks_id)):
            self.sam2_obj_counter += 1
            sam2_obj_id = self.sam2_obj_counter

            # Convert bbox to SAM2.1 format
            bbox_sam2 = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]], dtype=np.float32)

            # Use the last valid frame index from the predictor's image buffer
            # SAM2 camera predictor only keeps a limited buffer of recent images
            images = self.predictor.condition_state.get("images", [])
            frame_idx = len(images) - 1 if images else 0

            try:
                _, obj_ids, mask_logits = self.predictor.add_new_prompt(
                    frame_idx=frame_idx,
                    obj_id=sam2_obj_id,
                    bbox=bbox_sam2,
                )

                self.tracklet_to_sam2_obj[track_id] = sam2_obj_id

                # Update mask mappings
                next_mask_number = max_mask_number + i + 1
                new_masks_numbers.append(next_mask_number)
                self.tracklet_mask_dict[track_id] = next_mask_number

                self.mask_color_counter += 1
                self.mask_color_dict[next_mask_number] = self.mask_color_counter

            except RuntimeError as e:
                # If we can't add during tracking, defer to next opportunity
                print(f"Warning: Could not add mask for track {track_id}: {e}")
                self.awaiting_mask_tracklet_ids.append(track_id)

        self.num_objects = len(self.tracklet_mask_dict)

    def _remove_masks(self, removed_tracks_ids: List[int]):
        """Soft-remove masks for removed tracklets.

        SAM2.1 doesn't have a direct delete API, so we use soft removal
        (filtering in post-processing) to exclude removed objects.

        Parameters
        ----------
        removed_tracks_ids : List[int]
            IDs of tracklets to remove.
        """
        if len(removed_tracks_ids) == 0:
            return

        mask_ids_to_remove = []

        for track_id in removed_tracks_ids:
            if track_id in self.tracklet_to_sam2_obj:
                sam2_obj_id = self.tracklet_to_sam2_obj[track_id]
                self.removed_obj_ids.add(sam2_obj_id)

            if track_id in self.tracklet_mask_dict:
                mask_ids_to_remove.append(self.tracklet_mask_dict[track_id])

        # Update tracklet_mask_dict with renumbering
        self._update_mask_dicts_after_removal(mask_ids_to_remove, removed_tracks_ids)

    def _update_mask_dicts_after_removal(self, removed_mask_ids: List[int], removed_track_ids: List[int]):
        """Update mask dictionaries after removal with proper renumbering."""
        # Remove entries
        for track_id in removed_track_ids:
            self.tracklet_mask_dict.pop(track_id, None)

        # Renumber remaining mask IDs
        decrement_dict = {}
        for track_id, mask_id in self.tracklet_mask_dict.items():
            decrement = sum(1 for rmi in removed_mask_ids if mask_id > rmi)
            if decrement > 0:
                decrement_dict[track_id] = decrement

        for track_id, decrement in decrement_dict.items():
            self.tracklet_mask_dict[track_id] -= decrement

        # Update mask_color_dict similarly
        new_mask_color_dict = {}
        for mask_id, color_id in self.mask_color_dict.items():
            if mask_id not in removed_mask_ids:
                decrement = sum(1 for rmi in removed_mask_ids if mask_id > rmi)
                new_mask_color_dict[mask_id - decrement] = color_id
        self.mask_color_dict = new_mask_color_dict

        self.num_objects = len(self.tracklet_mask_dict)

    def _propagate_masks(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """Propagate masks to current frame.

        Parameters
        ----------
        frame : np.ndarray
            Current frame (BGR).

        Returns
        -------
        prediction : torch.Tensor or None
            Mask logits tensor of shape (num_objects, 1, H, W).
        """
        if not self.first_frame_loaded:
            return None

        # Track to current frame
        obj_ids, mask_logits = self.predictor.track(frame)
        self.frame_counter += 1

        # Combine masks, filtering out soft-removed objects
        return self._combine_masks(obj_ids, mask_logits, frame.shape[:2])

    def _combine_masks(
        self,
        obj_ids: List[int],
        mask_logits: torch.Tensor,
        image_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Combine per-object mask logits into a single prediction tensor.

        Parameters
        ----------
        obj_ids : List[int]
            List of SAM2.1 object IDs.
        mask_logits : torch.Tensor
            Mask logits of shape (num_objects, 1, H, W).
        image_shape : Tuple[int, int]
            Target (H, W) for output.

        Returns
        -------
        prediction : torch.Tensor
            Probability tensor of shape (num_objects+1, H, W).
        """
        H, W = image_shape
        num_objects = len(self.tracklet_mask_dict)

        # Initialize prediction tensor with background channel
        prediction = torch.zeros((num_objects + 1, H, W), device=self.device)

        for i, sam2_obj_id in enumerate(obj_ids):
            # Skip soft-removed objects
            if sam2_obj_id in self.removed_obj_ids:
                continue

            # Find corresponding track_id and mask_id
            track_id = None
            for tid, sid in self.tracklet_to_sam2_obj.items():
                if sid == sam2_obj_id:
                    track_id = tid
                    break

            if track_id is None or track_id not in self.tracklet_mask_dict:
                continue

            mask_id = self.tracklet_mask_dict[track_id]

            # Get mask logits for this object
            logits = mask_logits[i]  # Shape: (1, H, W)

            # Convert logits to probability
            prob = torch.sigmoid(logits.squeeze())

            # Resize if needed
            if prob.shape != (H, W):
                prob = F.interpolate(
                    prob[None, None],
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()

            if mask_id <= num_objects:
                prediction[mask_id] = prob

        # Background = 1 - max(foreground)
        if num_objects > 0:
            fg_max = prediction[1:].max(dim=0).values.clamp(0, 1)
            prediction[0] = 1.0 - fg_max
        else:
            prediction[0] = 1.0

        return prediction

    def _post_process_mask(
        self,
        prediction: torch.Tensor,
    ) -> Tuple[np.ndarray, Dict[int, float], np.ndarray]:
        """Post-process mask prediction.

        Parameters
        ----------
        prediction : torch.Tensor
            Probability tensor of shape (num_objects+1, H, W).

        Returns
        -------
        prediction_np : np.ndarray
            Integer mask of shape (H, W) with unique IDs.
        mask_avg_prob_dict : Dict[int, float]
            Average probability for each mask.
        prediction_colors : np.ndarray
            Color-preserved mask for visualization.
        """
        # Calculate average probabilities
        mask_avg_prob_dict = self._get_mask_avg_prob(prediction)

        # Convert to numpy mask via argmax
        prediction_np = prediction.argmax(dim=0).cpu().numpy().astype(np.uint8)

        # Store for next frame
        self.mask_prediction_prev_frame = prediction_np.copy()

        # Apply color preservation
        prediction_colors = self._adjust_mask_colors(prediction_np)

        return prediction_np, mask_avg_prob_dict, prediction_colors

    def _get_mask_avg_prob(self, prediction: torch.Tensor) -> Dict[int, float]:
        """Calculate average probability for each mask region.

        Parameters
        ----------
        prediction : torch.Tensor
            Probability tensor of shape (num_objects+1, H, W).

        Returns
        -------
        mask_avg_prob_dict : Dict[int, float]
            Average probability for each mask_id.
        """
        mask_avg_prob_dict = {}
        mask_maxes = torch.argmax(prediction, dim=0)

        for mask_id in self.tracklet_mask_dict.values():
            if mask_id < prediction.shape[0]:
                mask_region = mask_maxes == mask_id
                if mask_region.any():
                    avg_prob = prediction[mask_id][mask_region].mean().item()
                    if not np.isnan(avg_prob):
                        mask_avg_prob_dict[mask_id] = avg_prob

        return mask_avg_prob_dict

    def _adjust_mask_colors(self, prediction: np.ndarray) -> np.ndarray:
        """Preserve mask colors across removals.

        Parameters
        ----------
        prediction : np.ndarray
            Integer mask of shape (H, W).

        Returns
        -------
        prediction_colors : np.ndarray
            Color-adjusted mask.
        """
        prediction_colors = prediction.copy()

        # Apply color mapping in descending order to avoid conflicts
        keys_descending = sorted(self.mask_color_dict.keys(), reverse=True)
        for k in keys_descending:
            prediction_colors[prediction_colors == k] = self.mask_color_dict[k]

        return prediction_colors

    def _get_tracklets_with_lower_bottom(
        self,
        new_tracklet_tlwh: List[float],
        online_tlwhs: List,
    ) -> List:
        """Find tracklets with bottom edge below the given tracklet.

        Parameters
        ----------
        new_tracklet_tlwh : List[float]
            Target tracklet [top, left, width, height].
        online_tlwhs : List
            All current tracklet positions.

        Returns
        -------
        track_BBs_with_lower_bottom : List
            Tracklets with lower bottom edges.
        """
        nt_y = new_tracklet_tlwh[1]
        nt_h = new_tracklet_tlwh[3]
        nt_bottom = nt_y + nt_h

        track_BBs_with_lower_bottom = []
        for ot in online_tlwhs:
            if ot[1] + ot[3] > nt_bottom:
                track_BBs_with_lower_bottom.append(ot)

        return track_BBs_with_lower_bottom

    def _get_overlap_with_lower_bottom_tracklets(
        self,
        new_tracklet_tlwh: List[float],
        track_BBs_with_lower_bottom: List,
    ) -> float:
        """Calculate maximum overlap with lower tracklets.

        Parameters
        ----------
        new_tracklet_tlwh : List[float]
            Target tracklet [top, left, width, height].
        track_BBs_with_lower_bottom : List
            Tracklets with lower bottom edges.

        Returns
        -------
        max_overlap : float
            Maximum overlap ratio (0-1).
        """
        nt_x, nt_y, nt_w, nt_h = new_tracklet_tlwh

        max_overlap_part = 0.0

        for lb in track_BBs_with_lower_bottom:
            x_dist = min(nt_x + nt_w, lb[0] + lb[2]) - max(nt_x, lb[0])
            y_dist = min(nt_y + nt_h, lb[1] + lb[3]) - max(nt_y, lb[1])

            if x_dist < 0 or y_dist < 0:
                overlap_area = 0
            else:
                overlap_area = x_dist * y_dist

            bbox_area = nt_w * nt_h
            if bbox_area > 0:
                overlap_part = overlap_area / bbox_area
                if overlap_part > max_overlap_part:
                    max_overlap_part = overlap_part

                    if max_overlap_part >= 1:
                        break

        return max_overlap_part
