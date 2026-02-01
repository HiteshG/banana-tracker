"""BananaTracker: Multi-object tracking using YOLOv8 + ByteTrack + SAM2.1 + Cutie.

Main components:
- BananaTrackerConfig: Configuration dataclass
- BananaTrackerPipeline: Main pipeline (detector + tracker + visualizer + masks)
- BananaTracker: ByteTrack-based tracker with mask-enhanced cost matrix
- YOLOv8Detector: Detection wrapper
- MaskManager: SAM2.1 + Cutie for mask creation and temporal propagation

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
        output_video_path="output.mp4",
        # Enable mask module for enhanced tracking
        enable_masks=True,
        sam2_model_id="facebook/sam2.1-hiera-large",
        cutie_weights_path="/path/to/cutie-base-mega.pth",
    )

    pipeline = BananaTrackerPipeline(config)
    pipeline.process_video("input.mp4")
"""

__version__ = "0.2.0"

from .config import BananaTrackerConfig
from .detector import YOLOv8Detector
from .pipeline import BananaTrackerPipeline
from .visualizer import TrackVisualizer, VideoWriter, MOTWriter
from .tracker import BananaTracker, STrack, BaseTrack, TrackState, KalmanFilter

# Conditionally import mask module
try:
    from .mask_propagation import MaskManager
    _HAS_MASK_MODULE = True
except ImportError:
    MaskManager = None
    _HAS_MASK_MODULE = False

__all__ = [
    # Configuration
    "BananaTrackerConfig",

    # Main pipeline
    "BananaTrackerPipeline",

    # Components
    "YOLOv8Detector",
    "BananaTracker",
    "TrackVisualizer",
    "VideoWriter",
    "MOTWriter",

    # Track classes
    "STrack",
    "BaseTrack",
    "TrackState",
    "KalmanFilter",

    # Mask module
    "MaskManager",
]
