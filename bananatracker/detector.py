"""YOLOv8 detector wrapper with class filtering.

Supports:
- Class filtering (track_classes)
- Special classes with max-conf-only detection (e.g., single puck in hockey)
"""

import numpy as np
from typing import List, Optional, Tuple
from ultralytics import YOLO

from .config import BananaTrackerConfig


class YOLOv8Detector:
    """YOLOv8 object detector with class filtering support."""

    def __init__(self, config: BananaTrackerConfig):
        """Initialize the detector.

        Parameters
        ----------
        config : BananaTrackerConfig
            Configuration object with detector settings.
        """
        self.config = config
        self.model = YOLO(config.yolo_weights)

        # Move to device
        if config.device.startswith('cuda'):
            self.model.to(config.device)

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Run detection on a frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image frame.

        Returns
        -------
        detections : np.ndarray
            Detection array of shape (N, 6): [x1, y1, x2, y2, conf, class_id]
        """
        # Run inference
        results = self.model(frame, verbose=False, conf=self.config.detection_conf_thresh)

        if len(results) == 0 or results[0].boxes is None:
            return np.empty((0, 6))

        # Extract boxes
        boxes = results[0].boxes
        if len(boxes) == 0:
            return np.empty((0, 6))

        # Get detection data
        xyxy = boxes.xyxy.cpu().numpy()  # (N, 4)
        conf = boxes.conf.cpu().numpy()  # (N,)
        cls = boxes.cls.cpu().numpy()  # (N,)

        # Stack into (N, 6) array
        detections = np.column_stack([xyxy, conf, cls])

        # Apply class filtering
        detections = self._filter_by_track_classes(detections)

        # Apply special class filtering (max-conf only)
        detections = self._filter_special_classes(detections)

        return detections

    def _filter_by_track_classes(self, detections: np.ndarray) -> np.ndarray:
        """Filter detections to only include tracked classes.

        Parameters
        ----------
        detections : np.ndarray
            Detection array of shape (N, 6).

        Returns
        -------
        filtered : np.ndarray
            Filtered detection array.
        """
        if self.config.track_classes is None:
            return detections

        if len(detections) == 0:
            return detections

        # Get class IDs from column 5
        class_ids = detections[:, 5].astype(int)

        # Create mask for valid classes
        mask = np.isin(class_ids, self.config.track_classes)

        return detections[mask]

    def _filter_special_classes(self, detections: np.ndarray) -> np.ndarray:
        """Filter special classes to keep only max-confidence detection.

        For classes like 'Puck' in hockey, we typically only want to track
        a single instance, so we keep only the highest confidence detection.

        Parameters
        ----------
        detections : np.ndarray
            Detection array of shape (N, 6).

        Returns
        -------
        filtered : np.ndarray
            Filtered detection array.
        """
        if self.config.special_classes is None:
            return detections

        if len(detections) == 0:
            return detections

        result_dets = []
        class_ids = detections[:, 5].astype(int)

        # Process each special class
        for special_cls in self.config.special_classes:
            # Find all detections of this class
            cls_mask = class_ids == special_cls
            cls_dets = detections[cls_mask]

            if len(cls_dets) > 0:
                # Keep only the detection with highest confidence
                max_conf_idx = np.argmax(cls_dets[:, 4])
                result_dets.append(cls_dets[max_conf_idx:max_conf_idx + 1])

        # Add all non-special class detections
        non_special_mask = ~np.isin(class_ids, self.config.special_classes)
        non_special_dets = detections[non_special_mask]

        if len(non_special_dets) > 0:
            result_dets.append(non_special_dets)

        if len(result_dets) == 0:
            return np.empty((0, 6))

        return np.vstack(result_dets)

    def get_class_names(self) -> List[str]:
        """Get class names from the model.

        Returns
        -------
        names : List[str]
            List of class names.
        """
        return list(self.model.names.values())
