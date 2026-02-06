"""Configuration dataclass for BananaTracker."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class BananaTrackerConfig:
    """Configuration for the BananaTracker pipeline.

    Attributes:
        yolo_weights: Path to YOLOv8 .pt weights file
        class_names: List of class names in order ["Player", "Puck", "Referee", ...]
        track_classes: Class indices to track. None = all classes
        special_classes: Classes where only max-conf detection is kept (e.g., [1] for Puck)
        detection_conf_thresh: Minimum confidence for detections

        track_thresh: High-confidence detection threshold for first association
        track_buffer: Frames to keep lost tracks before removal (auto-scaled by video fps)
        match_thresh: Association matching threshold

        cmc_method: Camera motion compensation method ("orb", "ecc", "sift", "sparseOptFlow", "none")

        # Mask Module (SAM2.1 + Cutie)
        enable_masks: Whether to enable mask-based tracking enhancement
        sam2_model_id: HuggingFace model ID for SAM2.1
        sam2_checkpoint: Local checkpoint path for SAM2.1 (optional, overrides HF download)
        cutie_weights_path: Path to Cutie model weights
        hf_token: HuggingFace token for model access (optional)
        mask_start_frame: Frame to start mask processing (default 1)

        class_colors: Dict mapping class name to BGR color tuple
        show_track_id: Whether to display track IDs on visualization
        line_thickness: Bounding box line thickness

        output_video_path: Path to save output video (None = no video output)
        output_txt_path: Path to save MOT format results (None = no txt output)
        device: Device for inference ("cuda:0", "cpu", etc.)
    """

    # Detection
    yolo_weights: str = ""
    class_names: List[str] = field(default_factory=list)
    track_classes: Optional[List[int]] = None
    special_classes: Optional[List[int]] = None
    detection_conf_thresh: float = 0.4  # General confidence threshold (lowered from 0.5 to catch more objects)
    detection_iou_thresh: float = 0.7   # IoU threshold for YOLO NMS

    # Post-processing: Centroid-based deduplication (removes duplicate boxes for same object)
    centroid_dedup_enabled: bool = True
    centroid_dedup_max_distance: float = 36.0  # Max pixel distance to consider duplicates

    # Tracker (ByteTrack params)
    track_thresh: float = 0.5  # Lowered from 0.6 to match more detections in first pass
    track_buffer: int = 45     # Buffer in frames (actual duration depends on video fps)
    match_thresh: float = 0.8
    # Note: fps is auto-detected from video in pipeline.process_video()

    # Lost track recovery
    lost_track_buffer_scale: float = 0.3  # Expand bbox by 30% for lost track matching

    # Camera Motion Compensation ("orb", "ecc", "sift", "sparseOptFlow", "none")
    cmc_method: str = "ecc"  # ECC is better for fast motion/jitter (e.g., ice hockey)

    # Mask Module (SAM2.1 + Cutie)
    enable_masks: bool = False
    sam2_model_id: str = "facebook/sam2.1-hiera-large"
    sam2_checkpoint: Optional[str] = None  # Local checkpoint path (overrides HF download)
    cutie_weights_path: Optional[str] = None  # Path to cutie-base-mega.pth
    hf_token: Optional[str] = None  # HuggingFace token for gated models
    mask_start_frame: int = 1  # Frame to start mask processing
    mask_bbox_overlap_threshold: float = 0.6  # Threshold for avoiding overlapped mask creation

    # Visualization
    class_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    show_track_id: bool = True
    show_masks: bool = True  # Whether to show mask overlays when masks enabled
    mask_alpha: float = 0.5  # Transparency for mask overlay
    line_thickness: int = 2

    # Output
    output_video_path: Optional[str] = None
    output_txt_path: Optional[str] = None
    device: str = "cuda:0"

    # Debug
    debug_tracking: bool = False  # Enable debug logging for track lifecycle events

    def get_color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        """Get the BGR color for a given class ID."""
        if class_id < len(self.class_names):
            class_name = self.class_names[class_id]
            if class_name in self.class_colors:
                return self.class_colors[class_name]
        # Default color if not specified
        return (0, 255, 0)

    def get_class_name(self, class_id: int) -> str:
        """Get the class name for a given class ID."""
        if class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"
