"""Visualization utilities for track drawing and video output.

Features:
- Per-class color bounding boxes
- Track ID display with contrasting text color
- Mask overlay visualization with SAM2.1
- Video writer for output
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

from .config import BananaTrackerConfig


# Color palette for mask visualization (BGR format)
MASK_COLORS = [
    (255, 0, 0),     # Blue
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
    (255, 255, 0),   # Cyan
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (128, 0, 255),   # Purple
    (255, 128, 0),   # Orange
    (0, 128, 255),   # Light Orange
    (128, 255, 0),   # Lime
    (255, 0, 128),   # Pink
    (0, 255, 128),   # Spring Green
    (128, 128, 255), # Light Purple
    (128, 255, 255), # Light Cyan
    (255, 128, 128), # Light Blue
    (255, 255, 128), # Light Yellow
]


class TrackVisualizer:
    """Visualizer for drawing tracks on frames."""

    def __init__(self, config: BananaTrackerConfig):
        """Initialize the visualizer.

        Parameters
        ----------
        config : BananaTrackerConfig
            Configuration object with visualization settings.
        """
        self.config = config

    def draw_tracks(self, frame: np.ndarray, tracks: List) -> np.ndarray:
        """Draw bounding boxes and track IDs on frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image frame.
        tracks : List[STrack]
            List of active tracks.

        Returns
        -------
        frame : np.ndarray
            Frame with drawn tracks.
        """
        frame = frame.copy()

        for track in tracks:
            # Get bounding box
            x1, y1, x2, y2 = map(int, track.tlbr)

            # Get color for this class
            color = self.config.get_color_for_class(track.class_id)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.config.line_thickness)

            # Draw track ID
            if self.config.show_track_id:
                # Create label
                label = f"ID:{track.track_id}"
                if track.class_id >= 0 and track.class_id < len(self.config.class_names):
                    class_name = self.config.class_names[track.class_id]
                    label = f"{class_name} {track.track_id}"

                # Calculate text size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, font_thickness
                )

                # Draw text background
                text_x = x1
                text_y = y1 - 5
                if text_y < text_height + 5:
                    text_y = y1 + text_height + 5

                bg_x1 = text_x
                bg_y1 = text_y - text_height - 2
                bg_x2 = text_x + text_width + 4
                bg_y2 = text_y + 2

                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

                # Choose contrasting text color
                text_color = self._get_contrasting_color(color)

                # Draw text
                cv2.putText(
                    frame, label, (text_x + 2, text_y - 2),
                    font, font_scale, text_color, font_thickness
                )

        return frame

    def draw_masks(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        tracklet_mask_dict: Optional[Dict[int, int]] = None,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Draw mask overlays on frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image frame.
        mask : np.ndarray
            Integer mask of shape (H, W) where 0=background and >0 are object IDs.
        tracklet_mask_dict : Dict[int, int], optional
            Mapping from track_id to mask_id for consistent coloring.
        alpha : float, optional
            Transparency of mask overlay (0-1). Defaults to config.mask_alpha.

        Returns
        -------
        frame : np.ndarray
            Frame with mask overlays.
        """
        frame = frame.copy()

        if mask is None or mask.max() == 0:
            return frame

        if alpha is None:
            alpha = self.config.mask_alpha

        # Create colored overlay
        overlay = np.zeros_like(frame)

        # Get unique mask IDs (excluding background)
        unique_ids = np.unique(mask)
        unique_ids = unique_ids[unique_ids > 0]

        for mask_id in unique_ids:
            # Get color for this mask
            color_idx = (mask_id - 1) % len(MASK_COLORS)
            color = MASK_COLORS[color_idx]

            # Create mask region
            mask_region = mask == mask_id
            overlay[mask_region] = color

        # Blend overlay with frame
        mask_any = mask > 0
        frame[mask_any] = cv2.addWeighted(
            frame[mask_any], 1 - alpha,
            overlay[mask_any], alpha,
            0
        )

        return frame

    def draw_tracks_with_masks(
        self,
        frame: np.ndarray,
        tracks: List,
        mask: Optional[np.ndarray] = None,
        tracklet_mask_dict: Optional[Dict[int, int]] = None,
    ) -> np.ndarray:
        """Draw both masks and tracks on frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image frame.
        tracks : List[STrack]
            List of active tracks.
        mask : np.ndarray, optional
            Integer mask of shape (H, W).
        tracklet_mask_dict : Dict[int, int], optional
            Mapping from track_id to mask_id.

        Returns
        -------
        frame : np.ndarray
            Frame with masks and tracks drawn.
        """
        # Draw masks first (underneath)
        if mask is not None:
            frame = self.draw_masks(frame, mask, tracklet_mask_dict)

        # Draw tracks on top
        frame = self.draw_tracks(frame, tracks)

        return frame

    def draw_detections(self, frame: np.ndarray, detections: np.ndarray,
                        color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draw raw detections on frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image frame.
        detections : np.ndarray
            Detection array of shape (N, 5+).
        color : Tuple[int, int, int]
            BGR color for detections.

        Returns
        -------
        frame : np.ndarray
            Frame with drawn detections.
        """
        frame = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            conf = det[4] if len(det) > 4 else 0

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            if conf > 0:
                label = f"{conf:.2f}"
                cv2.putText(
                    frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
                )

        return frame

    @staticmethod
    def _get_contrasting_color(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Get a contrasting color (black or white) for text readability.

        Parameters
        ----------
        color : Tuple[int, int, int]
            BGR background color.

        Returns
        -------
        text_color : Tuple[int, int, int]
            BGR color for text (black or white).
        """
        # Calculate luminance (BGR order)
        b, g, r = color
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255

        # Return black for bright backgrounds, white for dark
        if luminance > 0.5:
            return (0, 0, 0)  # Black
        else:
            return (255, 255, 255)  # White


class VideoWriter:
    """Video writer for saving tracked output."""

    def __init__(self, output_path: str, fps: int = 30, frame_size: Optional[Tuple[int, int]] = None):
        """Initialize the video writer.

        Parameters
        ----------
        output_path : str
            Path to save the output video.
        fps : int
            Frames per second.
        frame_size : Tuple[int, int], optional
            (width, height) of the video. If None, set from first frame.
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None
        self.initialized = False

    def write(self, frame: np.ndarray):
        """Write a frame to the video.

        Parameters
        ----------
        frame : np.ndarray
            BGR image frame.
        """
        if not self.initialized:
            height, width = frame.shape[:2]
            self.frame_size = (width, height)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, self.frame_size
            )
            self.initialized = True

        self.writer.write(frame)

    def release(self):
        """Release the video writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


class MOTWriter:
    """Writer for MOT format results."""

    def __init__(self, output_path: str):
        """Initialize the MOT writer.

        Parameters
        ----------
        output_path : str
            Path to save the MOT format results.
        """
        self.output_path = output_path
        self.file = None

    def open(self):
        """Open the output file."""
        self.file = open(self.output_path, 'w')

    def write_frame(self, frame_id: int, tracks: List):
        """Write tracks for a frame in MOT format.

        MOT format: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        Parameters
        ----------
        frame_id : int
            Frame number (1-indexed).
        tracks : List[STrack]
            List of active tracks.
        """
        if self.file is None:
            self.open()

        for track in tracks:
            tlwh = track.tlwh
            line = f"{frame_id},{track.track_id},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{track.score:.2f},-1,-1,-1\n"
            self.file.write(line)

    def close(self):
        """Close the output file."""
        if self.file is not None:
            self.file.close()
            self.file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
