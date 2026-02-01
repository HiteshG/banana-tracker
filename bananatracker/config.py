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
        track_buffer: Frames to keep lost tracks before removal
        match_thresh: Association matching threshold
        fps: Video frame rate (used for buffer calculation)

        cmc_method: Camera motion compensation method ("orb", "ecc", "sift", "sparseOptFlow", "none")

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
    detection_conf_thresh: float = 0.5  # General confidence threshold
    detection_iou_thresh: float = 0.7   # IoU threshold for YOLO NMS

    # Post-processing: Centroid-based deduplication (removes duplicate boxes for same object)
    centroid_dedup_enabled: bool = True
    centroid_dedup_max_distance: float = 36.0  # Max pixel distance to consider duplicates

    # Tracker (ByteTrack params)
    track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8
    fps: int = 30

    # Camera Motion Compensation
    cmc_method: str = "orb"

    # SAM2.1 Mask Propagation
    sam2_enabled: bool = True  # Enable mask generation with SAM2.1
    sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt"  # SAM2.1 model checkpoint
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml"  # SAM2.1 config yaml
    sam2_repo_path: str = ""  # Path to segment-anything-2-real-time repo (auto-detected if empty)
    mask_start_frame: int = 1  # Frame to start mask creation (1-indexed)
    mask_overlap_threshold: float = 0.6  # Skip mask creation for heavily overlapping bboxes
    mask_alpha: float = 0.4  # Mask overlay transparency for visualization

    # Visualization
    class_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    show_track_id: bool = True
    line_thickness: int = 2

    # Output
    output_video_path: Optional[str] = None
    output_txt_path: Optional[str] = None
    device: str = "cuda:0"

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
