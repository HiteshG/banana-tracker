"""Main pipeline combining detector, tracker, and visualizer.

Usage:
    from bananatracker import BananaTrackerConfig, BananaTrackerPipeline

    config = BananaTrackerConfig(...)
    pipeline = BananaTrackerPipeline(config)
    pipeline.process_video("input.mp4")
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Generator
from tqdm import tqdm

from .config import BananaTrackerConfig
from .detector import YOLOv8Detector
from .tracker import BananaTracker
from .visualizer import TrackVisualizer, VideoWriter, MOTWriter


class BananaTrackerPipeline:
    """Main tracking pipeline combining detection, tracking, and visualization."""

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
        self.tracker = BananaTracker(
            track_thresh=config.track_thresh,
            track_buffer=config.track_buffer,
            match_thresh=config.match_thresh,
            frame_rate=config.fps,
            cmc_method=config.cmc_method
        )
        self.visualizer = TrackVisualizer(config)

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

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Update fps in tracker if different
        if fps != self.config.fps:
            self.tracker.buffer_size = int(fps / 30.0 * self.config.track_buffer)
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

                # Update tracker
                img_info = (height, width)
                tracks, removed_ids, new_tracks = self.tracker.update(
                    detections_array=detections,
                    img_info=img_info,
                    frame_img=frame
                )

                # Store tracks
                all_tracks.append((frame_id, tracks))

                # Write MOT format
                if mot_writer:
                    mot_writer.write_frame(frame_id, tracks)

                # Draw visualization
                if video_writer:
                    vis_frame = self.visualizer.draw_tracks(frame, tracks)
                    video_writer.write(vis_frame)

        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            if mot_writer:
                mot_writer.close()

        return all_tracks

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

        # Update tracker
        img_info = (height, width)
        tracks, removed_ids, new_tracks = self.tracker.update(
            detections_array=detections,
            img_info=img_info,
            frame_img=frame
        )

        # Draw visualization
        vis_frame = self.visualizer.draw_tracks(frame, tracks)

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
            Frame with drawn tracks.
        """
        # Reset tracker
        self.tracker.reset()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Update tracker buffer if needed
        if fps != self.config.fps:
            self.tracker.buffer_size = int(fps / 30.0 * self.config.track_buffer)
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

                # Track
                img_info = (height, width)
                tracks, removed_ids, new_tracks = self.tracker.update(
                    detections_array=detections,
                    img_info=img_info,
                    frame_img=frame
                )

                # Visualize
                vis_frame = self.visualizer.draw_tracks(frame, tracks)

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
