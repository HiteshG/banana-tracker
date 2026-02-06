"""Main pipeline combining detector, tracker, visualizer, and mask module.

Usage:
    from bananatracker import BananaTrackerConfig, BananaTrackerPipeline

    config = BananaTrackerConfig(...)
    pipeline = BananaTrackerPipeline(config)
    pipeline.process_video("input.mp4")
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Generator, Dict
from tqdm import tqdm

from .config import BananaTrackerConfig
from .detector import YOLOv8Detector
from .tracker import BananaTracker
from .visualizer import TrackVisualizer, VideoWriter, MOTWriter


class BananaTrackerPipeline:
    """Main tracking pipeline combining detection, tracking, visualization, and masks."""

    def __init__(self, config: BananaTrackerConfig):
        """Initialize the pipeline.

        Parameters
        ----------
        config : BananaTrackerConfig
            Configuration object with all settings.
        """
        self.config = config

        # Initialize components
        self.detector = YOLOv8Detector(config)
        # Initialize tracker with default 30fps - will be updated when video opens
        self.tracker = BananaTracker(
            track_thresh=config.track_thresh,
            track_buffer=config.track_buffer,
            match_thresh=config.match_thresh,
            frame_rate=30,  # Default, updated dynamically from video
            cmc_method=config.cmc_method
        )
        self.visualizer = TrackVisualizer(config)

        # Initialize mask module if enabled
        self.mask_manager = None
        if config.enable_masks:
            try:
                from .mask_propagation import MaskManager
                self.mask_manager = MaskManager(
                    sam2_model_id=config.sam2_model_id,
                    sam2_checkpoint=config.sam2_checkpoint,
                    cutie_weights_path=config.cutie_weights_path,
                    device=config.device,
                    hf_token=config.hf_token,
                    mask_start_frame=config.mask_start_frame,
                    bbox_overlap_threshold=config.mask_bbox_overlap_threshold
                )
                print("Mask module (SAM2.1 + Cutie) initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize mask module: {e}")
                print("Proceeding without mask-enhanced tracking")
                self.mask_manager = None

        # Mask state tracking
        self.prediction_mask = None
        self.tracklet_mask_dict = {}
        self.mask_avg_prob_dict = {}
        self.mask_colors = None
        self.prev_frame_info = None

    def process_video(self, video_path: str, show_progress: bool = True) -> List:
        """Process a video file for tracking.

        Parameters
        ----------
        video_path : str
            Path to input video file.
        show_progress : bool
            Whether to show progress bar.

        Returns
        -------
        all_tracks : List
            List of (frame_id, tracks) tuples for all frames.
        """
        # Reset tracker for new video
        self.tracker.reset()
        self._reset_mask_state()

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Update buffer size based on actual video fps, capped at 45 frames
        self.tracker.buffer_size = min(int(fps / 30.0 * self.config.track_buffer), 45)
        self.tracker.max_time_lost = self.tracker.buffer_size

        # Setup output writers
        video_writer = None
        mot_writer = None

        if self.config.output_video_path:
            video_writer = VideoWriter(self.config.output_video_path, fps=fps)

        if self.config.output_txt_path:
            mot_writer = MOTWriter(self.config.output_txt_path)
            mot_writer.open()

        # Process frames
        all_tracks = []
        frame_id = 0

        iterator = range(total_frames)
        if show_progress:
            iterator = tqdm(iterator, desc="Processing", unit="frame")

        try:
            for _ in iterator:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_id += 1

                # Detect objects
                detections = self.detector.detect(frame)

                # Update tracker with mask parameters
                img_info = (height, width)
                tracks, removed_ids, new_tracks = self.tracker.update(
                    detections_array=detections,
                    img_info=img_info,
                    prediction_mask=self.prediction_mask,
                    tracklet_mask_dict=self.tracklet_mask_dict,
                    mask_avg_prob_dict=self.mask_avg_prob_dict,
                    frame_img=frame
                )

                # Update masks if enabled
                if self.mask_manager is not None:
                    self._update_masks(frame, frame_id, tracks, new_tracks, removed_ids)

                # Store tracks
                all_tracks.append((frame_id, tracks))

                # Write MOT format
                if mot_writer:
                    mot_writer.write_frame(frame_id, tracks)

                # Draw visualization with optional mask overlay
                if video_writer:
                    vis_frame = self._draw_with_masks(frame, tracks)
                    video_writer.write(vis_frame)

        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            if mot_writer:
                mot_writer.close()

        return all_tracks

    def _reset_mask_state(self):
        """Reset mask-related state for new video."""
        self.prediction_mask = None
        self.tracklet_mask_dict = {}
        self.mask_avg_prob_dict = {}
        self.mask_colors = None
        self.prev_frame_info = None

    def _update_masks(self, frame: np.ndarray, frame_id: int,
                      tracks: List, new_tracks: List, removed_ids: List[int]):
        """Update masks using MaskManager.

        Parameters
        ----------
        frame : np.ndarray
            Current BGR frame
        frame_id : int
            Current frame number
        tracks : List
            Active tracks
        new_tracks : List
            Newly created tracks
        removed_ids : List[int]
            IDs of removed tracks
        """
        # Prepare frame info
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_info = {'raw_img': frame_rgb}

        # Use previous frame info (or current if first frame)
        if self.prev_frame_info is None:
            self.prev_frame_info = img_info

        # Get track positions and IDs
        online_tlwhs = [track.tlwh for track in tracks]
        online_ids = [track.track_id for track in tracks]

        # Update masks
        self.prediction_mask, self.tracklet_mask_dict, self.mask_avg_prob_dict, self.mask_colors = \
            self.mask_manager.get_updated_masks(
                img_info=img_info,
                img_info_prev=self.prev_frame_info,
                frame_id=frame_id,
                online_tlwhs=online_tlwhs,
                online_ids=online_ids,
                new_tracks=new_tracks,
                removed_tracks_ids=removed_ids
            )

        # Store current frame for next iteration
        self.prev_frame_info = img_info

    def _draw_with_masks(self, frame: np.ndarray, tracks: List) -> np.ndarray:
        """Draw tracks with optional mask overlay.

        Parameters
        ----------
        frame : np.ndarray
            Original BGR frame
        tracks : List
            Active tracks

        Returns
        -------
        vis_frame : np.ndarray
            Frame with drawn tracks and optional masks
        """
        vis_frame = frame.copy()

        # Draw mask overlay if enabled and available
        if (self.config.show_masks and self.mask_colors is not None and
            self.mask_manager is not None):
            vis_frame = self._overlay_masks(vis_frame, self.mask_colors)

        # Draw track boxes and IDs
        vis_frame = self.visualizer.draw_tracks(vis_frame, tracks)

        return vis_frame

    def _overlay_masks(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Overlay colored masks on frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame
        mask : np.ndarray
            Mask array (H, W) with unique values per object

        Returns
        -------
        overlayed : np.ndarray
            Frame with mask overlay
        """
        if mask is None:
            return frame

        # Color palette for masks (DAVIS palette style)
        colors = [
            (0, 0, 0),       # Background
            (128, 0, 0),     # Object 1
            (0, 128, 0),     # Object 2
            (128, 128, 0),   # Object 3
            (0, 0, 128),     # Object 4
            (128, 0, 128),   # Object 5
            (0, 128, 128),   # Object 6
            (128, 128, 128), # Object 7
            (64, 0, 0),      # Object 8
            (192, 0, 0),     # Object 9
            (64, 128, 0),    # Object 10
        ]

        # Create colored mask
        H, W = mask.shape
        colored_mask = np.zeros((H, W, 3), dtype=np.uint8)

        unique_ids = np.unique(mask)
        for obj_id in unique_ids:
            if obj_id == 0:
                continue
            color_idx = obj_id % len(colors)
            if color_idx == 0:
                color_idx = 1
            colored_mask[mask == obj_id] = colors[color_idx]

        # Blend with original frame
        alpha = self.config.mask_alpha
        binary_mask = (mask > 0)
        overlayed = frame.copy()
        foreground = frame * alpha + colored_mask * (1 - alpha)
        overlayed[binary_mask] = foreground[binary_mask].astype(np.uint8)

        return overlayed

    def process_frame(self, frame: np.ndarray, frame_id: int = None) -> Tuple[List, np.ndarray]:
        """Process a single frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image frame.
        frame_id : int, optional
            Frame number (auto-incremented if None).

        Returns
        -------
        tracks : List[STrack]
            List of active tracks.
        vis_frame : np.ndarray
            Frame with drawn tracks.
        """
        height, width = frame.shape[:2]

        # Detect objects
        detections = self.detector.detect(frame)

        # Update tracker with mask parameters
        img_info = (height, width)
        tracks, removed_ids, new_tracks = self.tracker.update(
            detections_array=detections,
            img_info=img_info,
            prediction_mask=self.prediction_mask,
            tracklet_mask_dict=self.tracklet_mask_dict,
            mask_avg_prob_dict=self.mask_avg_prob_dict,
            frame_img=frame
        )

        # Update masks if enabled
        if self.mask_manager is not None and frame_id is not None:
            self._update_masks(frame, frame_id, tracks, new_tracks, removed_ids)

        # Draw visualization with masks
        vis_frame = self._draw_with_masks(frame, tracks)

        return tracks, vis_frame

    def process_video_generator(self, video_path: str) -> Generator:
        """Process video as a generator yielding frame-by-frame results.

        Parameters
        ----------
        video_path : str
            Path to input video file.

        Yields
        ------
        frame_id : int
            Frame number.
        frame : np.ndarray
            Original frame.
        tracks : List[STrack]
            Active tracks for this frame.
        vis_frame : np.ndarray
            Frame with drawn tracks (and masks if enabled).
        """
        # Reset tracker and mask state
        self.tracker.reset()
        self._reset_mask_state()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Update buffer size based on actual video fps, capped at 45 frames
        self.tracker.buffer_size = min(int(fps / 30.0 * self.config.track_buffer), 45)
        self.tracker.max_time_lost = self.tracker.buffer_size

        frame_id = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_id += 1

                # Detect
                detections = self.detector.detect(frame)

                # Track with mask parameters
                img_info = (height, width)
                tracks, removed_ids, new_tracks = self.tracker.update(
                    detections_array=detections,
                    img_info=img_info,
                    prediction_mask=self.prediction_mask,
                    tracklet_mask_dict=self.tracklet_mask_dict,
                    mask_avg_prob_dict=self.mask_avg_prob_dict,
                    frame_img=frame
                )

                # Update masks if enabled
                if self.mask_manager is not None:
                    self._update_masks(frame, frame_id, tracks, new_tracks, removed_ids)

                # Visualize with masks
                vis_frame = self._draw_with_masks(frame, tracks)

                yield frame_id, frame, tracks, vis_frame

        finally:
            cap.release()

    def get_track_info(self, tracks: List) -> List[dict]:
        """Convert tracks to dictionary format.

        Parameters
        ----------
        tracks : List[STrack]
            List of active tracks.

        Returns
        -------
        track_info : List[dict]
            List of track dictionaries with id, bbox, class_id, score.
        """
        track_info = []
        for track in tracks:
            info = {
                'track_id': track.track_id,
                'bbox': track.tlbr.tolist(),  # [x1, y1, x2, y2]
                'tlwh': track.tlwh.tolist(),  # [top, left, width, height]
                'class_id': track.class_id,
                'class_name': self.config.get_class_name(track.class_id),
                'score': track.score
            }
            track_info.append(info)
        return track_info
