"""BananaTracker: Multi-object tracking with SAM2.1 mask propagation.

Main components:
- BananaTrackerConfig: Configuration dataclass
- BananaTrackerPipeline: Main pipeline (detector + tracker + mask manager + visualizer)
- BananaTracker: ByteTrack-based tracker
- MaskManager: SAM2.1-based mask generation and propagation
- YOLOv8Detector: Detection wrapper

Example usage:
    from bananatracker import BananaTrackerConfig, BananaTrackerPipeline

    config = BananaTrackerConfig(
        yolo_weights="model.pt",
        class_names=["Player", "Puck", "Referee"],
        track_classes=[0, 1, 2],
        class_colors={
            "Player": (255, 0, 0),
            "Puck": (0, 255, 0),
            "Referee": (0, 0, 255),
        },
        # SAM2.1 configuration
        sam2_enabled=True,
        sam2_checkpoint="checkpoints/sam2.1_hiera_large.pt",
        sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml",
        output_video_path="output.mp4",
    )

    pipeline = BananaTrackerPipeline(config)
    pipeline.process_video("input.mp4")
"""

__version__ = "0.1.0"

from .config import BananaTrackerConfig
from .detector import YOLOv8Detector
from .pipeline import BananaTrackerPipeline
from .visualizer import TrackVisualizer, VideoWriter, MOTWriter
from .tracker import BananaTracker, STrack, BaseTrack, TrackState, KalmanFilter
from .mask_manager import MaskManager

__all__ = [
    # Configuration
    "BananaTrackerConfig",

    # Main pipeline
    "BananaTrackerPipeline",

    # Components
    "YOLOv8Detector",
    "BananaTracker",
    "MaskManager",
    "TrackVisualizer",
    "VideoWriter",
    "MOTWriter",

    # Track classes
    "STrack",
    "BaseTrack",
    "TrackState",
    "KalmanFilter",
]
