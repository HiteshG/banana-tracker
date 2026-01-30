"""Camera Motion Compensation (GMC) module.

Ported from McByte and enhanced with BoxMOT improvements:
- Detection masking: excludes detected objects from keypoint matching
- Lowe's ratio test: better match filtering (0.9 threshold)
- Spatial gating: rejects matches with large spatial distance (2.5 sigma)
"""

import copy
import numpy as np
import cv2
from typing import Optional


class GMC:
    """Global Motion Compensation class.

    Supports multiple methods for camera motion estimation:
    - orb: ORB keypoint detection + RANSAC (fast, robust)
    - sift: SIFT keypoint detection (slower, more accurate)
    - ecc: Enhanced Correlation Coefficient (accurate for subtle motion)
    - sparseOptFlow: Lucas-Kanade sparse optical flow
    - none: No compensation (identity matrix)
    """

    def __init__(self, method: str = 'orb', downscale: int = 2, verbose=None):
        """Initialize GMC.

        Parameters
        ----------
        method : str
            Method for motion estimation: 'orb', 'sift', 'ecc', 'sparseOptFlow', 'none'
        downscale : int
            Downscale factor for processing (higher = faster, less accurate)
        verbose : optional
            Additional parameters for 'file' method
        """
        self.method = method
        self.downscale = max(1, int(downscale))

        if self.method == 'orb':
            self.detector = cv2.FastFeatureDetector_create(20)
            self.extractor = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        elif self.method == 'sift':
            self.detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.extractor = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        elif self.method == 'ecc':
            number_of_iterations = 5000
            termination_eps = 1e-6
            self.warp_mode = cv2.MOTION_EUCLIDEAN
            self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        elif self.method == 'sparseOptFlow':
            self.feature_params = dict(
                maxCorners=1000,
                qualityLevel=0.01,
                minDistance=1,
                blockSize=3,
                useHarrisDetector=False,
                k=0.04
            )

        elif self.method == 'none' or self.method == 'None':
            self.method = 'none'

        else:
            raise ValueError(f"Unknown CMC method: {method}")

        self.prevFrame = None
        self.prevKeyPoints = None
        self.prevDescriptors = None

        self.initializedFirstFrame = False

    def apply(self, raw_frame: np.ndarray, detections: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply motion compensation.

        Parameters
        ----------
        raw_frame : np.ndarray
            BGR image frame.
        detections : np.ndarray, optional
            Detection boxes in tlbr format (N, 4+).

        Returns
        -------
        H : np.ndarray
            2x3 affine transformation matrix.
        """
        if self.method == 'orb' or self.method == 'sift':
            return self.applyFeatures(raw_frame, detections)
        elif self.method == 'ecc':
            return self.applyEcc(raw_frame, detections)
        elif self.method == 'sparseOptFlow':
            return self.applySparseOptFlow(raw_frame, detections)
        elif self.method == 'none':
            return np.eye(2, 3)
        else:
            return np.eye(2, 3)

    def _generate_mask(self, frame: np.ndarray, detections: Optional[np.ndarray]) -> np.ndarray:
        """Generate mask for keypoint detection.

        Creates a mask that:
        - Keeps a central safe region (removes noisy borders)
        - Excludes detected dynamic objects

        Parameters
        ----------
        frame : np.ndarray
            Grayscale frame (after downscaling).
        detections : np.ndarray, optional
            Detection boxes in tlbr format (original scale).

        Returns
        -------
        mask : np.ndarray
            Binary mask (255 = valid, 0 = exclude).
        """
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        # Keep most of the image, drop extreme borders
        y1, y2 = int(0.02 * height), int(0.98 * height)
        x1, x2 = int(0.02 * width), int(0.98 * width)
        mask[y1:y2, x1:x2] = 255

        # Exclude detection regions
        if detections is not None and len(detections) > 0:
            for det in detections:
                if len(det) < 4:
                    continue
                tlbr = (det[:4] / self.downscale).astype(np.int_)
                x1b = max(0, min(width, tlbr[0]))
                x2b = max(0, min(width, tlbr[2]))
                y1b = max(0, min(height, tlbr[1]))
                y2b = max(0, min(height, tlbr[3]))
                if x2b > x1b and y2b > y1b:
                    mask[y1b:y2b, x1b:x2b] = 0

        return mask

    def applyEcc(self, raw_frame: np.ndarray, detections: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply ECC-based motion compensation."""
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3, dtype=np.float32)

        if self.downscale > 1.0:
            frame = cv2.GaussianBlur(frame, (3, 3), 1.5)
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        # Handle first frame
        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.initializedFirstFrame = True
            return H

        # Run the ECC algorithm
        try:
            (cc, H) = cv2.findTransformECC(self.prevFrame, frame, H, self.warp_mode, self.criteria, None, 1)
        except Exception:
            pass  # Return identity on failure

        self.prevFrame = frame.copy()
        return H

    def applyFeatures(self, raw_frame: np.ndarray, detections: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply feature-based motion compensation (ORB/SIFT).

        Enhanced with BoxMOT improvements:
        - Detection masking
        - Lowe's ratio test (0.9)
        - Spatial gating (2.5 sigma)
        """
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        if self.downscale > 1.0:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))
            width = width // self.downscale
            height = height // self.downscale

        # Generate mask excluding detected objects (BoxMOT enhancement)
        mask = self._generate_mask(frame, detections)

        # Detect keypoints with mask
        keypoints = self.detector.detect(frame, mask)

        # Compute descriptors
        keypoints, descriptors = self.extractor.compute(frame, keypoints)

        # Handle first frame
        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            self.initializedFirstFrame = True
            return H

        # Check for valid descriptors
        if descriptors is None or len(keypoints) < 4:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            return H

        if self.prevDescriptors is None or self.prevKeyPoints is None:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            return H

        # Match descriptors using KNN
        knnMatches = self.matcher.knnMatch(self.prevDescriptors, descriptors, 2)

        # Handle empty matches
        if len(knnMatches) == 0:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            return H

        # Lowe's ratio test + spatial gating (BoxMOT enhancement)
        matches = []
        spatialDistances = []
        maxSpatialDistance = 0.25 * np.array([width, height])

        for pair in knnMatches:
            if len(pair) != 2:
                continue
            m, n = pair

            # Lowe's ratio test (0.9 threshold)
            if m.distance >= 0.9 * n.distance:
                continue

            prevKeyPointLocation = self.prevKeyPoints[m.queryIdx].pt
            currKeyPointLocation = keypoints[m.trainIdx].pt

            spatialDistance = (
                prevKeyPointLocation[0] - currKeyPointLocation[0],
                prevKeyPointLocation[1] - currKeyPointLocation[1]
            )

            # Spatial gating: reject large displacements
            if (abs(spatialDistance[0]) < maxSpatialDistance[0]) and \
                    (abs(spatialDistance[1]) < maxSpatialDistance[1]):
                spatialDistances.append(spatialDistance)
                matches.append(m)

        if len(matches) < 4:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.prevDescriptors = copy.copy(descriptors)
            return H

        # Statistical outlier rejection (2.5 sigma) - BoxMOT enhancement
        spatialDistances = np.asarray(spatialDistances, dtype=np.float32)
        meanSpatialDistances = spatialDistances.mean(axis=0)
        stdSpatialDistances = spatialDistances.std(axis=0) + 1e-6

        inliers = np.all((spatialDistances - meanSpatialDistances) < 2.5 * stdSpatialDistances, axis=1)

        goodMatches = []
        prevPoints = []
        currPoints = []
        for i in range(len(matches)):
            if inliers[i]:
                goodMatches.append(matches[i])
                prevPoints.append(self.prevKeyPoints[matches[i].queryIdx].pt)
                currPoints.append(keypoints[matches[i].trainIdx].pt)

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Estimate affine transformation
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(currPoints, 0)):
            H, inliers_mask = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            if H is not None:
                # Handle downscale
                if self.downscale > 1.0:
                    H[0, 2] *= self.downscale
                    H[1, 2] *= self.downscale
            else:
                H = np.eye(2, 3)
        else:
            H = np.eye(2, 3)

        # Store for next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)
        self.prevDescriptors = copy.copy(descriptors)

        return H

    def applySparseOptFlow(self, raw_frame: np.ndarray, detections: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply sparse optical flow-based motion compensation."""
        height, width, _ = raw_frame.shape
        frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(2, 3)

        # Downscale image
        if self.downscale > 1.0:
            frame = cv2.resize(frame, (width // self.downscale, height // self.downscale))

        # Find keypoints
        keypoints = cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

        # Handle first frame
        if not self.initializedFirstFrame:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            self.initializedFirstFrame = True
            return H

        if keypoints is None or self.prevKeyPoints is None:
            self.prevFrame = frame.copy()
            self.prevKeyPoints = copy.copy(keypoints)
            return H

        # Find correspondences
        matchedKeypoints, status, err = cv2.calcOpticalFlowPyrLK(
            self.prevFrame, frame, self.prevKeyPoints, None
        )

        # Leave good correspondences only
        prevPoints = []
        currPoints = []

        for i in range(len(status)):
            if status[i]:
                prevPoints.append(self.prevKeyPoints[i])
                currPoints.append(matchedKeypoints[i])

        prevPoints = np.array(prevPoints)
        currPoints = np.array(currPoints)

        # Find rigid matrix
        if (np.size(prevPoints, 0) > 4) and (np.size(prevPoints, 0) == np.size(currPoints, 0)):
            H, inliers = cv2.estimateAffinePartial2D(prevPoints, currPoints, cv2.RANSAC)

            if H is not None:
                # Handle downscale
                if self.downscale > 1.0:
                    H[0, 2] *= self.downscale
                    H[1, 2] *= self.downscale
            else:
                H = np.eye(2, 3)

        # Store to next iteration
        self.prevFrame = frame.copy()
        self.prevKeyPoints = copy.copy(keypoints)

        return H
