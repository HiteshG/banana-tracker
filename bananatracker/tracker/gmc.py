"""Camera Motion Compensation (GMC) module.

Ported from McByte and enhanced with BoxMOT improvements:
- Detection masking: excludes detected objects from keypoint matching
- Lowe's ratio test: better match filtering (0.9 threshold)
- Spatial gating: rejects matches with large spatial distance (2.5 sigma)
- ECC: Reimplemented following BoxMOT's optimized pattern
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

    def __init__(
        self,
        method: str = 'orb',
        downscale: int = 2,
        verbose=None,
        # ECC-specific parameters (following BoxMOT pattern)
        ecc_max_iterations: int = 100,
        ecc_eps: float = 1e-5,
        ecc_scale: float = 0.25,  # 4x downscale
        ecc_warp_mode: int = cv2.MOTION_EUCLIDEAN,
        grayscale: bool = True
    ):
        """Initialize GMC.

        Parameters
        ----------
        method : str
            Method for motion estimation: 'orb', 'sift', 'ecc', 'sparseOptFlow', 'none'
        downscale : int
            Downscale factor for ORB/SIFT/sparseOptFlow (higher = faster, less accurate)
        verbose : optional
            Additional parameters for 'file' method
        ecc_max_iterations : int
            Maximum iterations for ECC algorithm (default: 100)
        ecc_eps : float
            Termination epsilon for ECC (default: 1e-5)
        ecc_scale : float
            Scale factor for ECC (0.25 = 4x downscale, 0.15 = ~6.6x downscale)
        ecc_warp_mode : int
            OpenCV warp mode for ECC (MOTION_TRANSLATION, MOTION_EUCLIDEAN, MOTION_AFFINE, MOTION_HOMOGRAPHY)
        grayscale : bool
            Whether to convert to grayscale for ECC
        """
        self.method = method
        self.downscale = max(1, int(downscale))
        self.grayscale = grayscale

        if self.method == 'orb':
            self.detector = cv2.FastFeatureDetector_create(20)
            self.extractor = cv2.ORB_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        elif self.method == 'sift':
            self.detector = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.extractor = cv2.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=20)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)

        elif self.method == 'ecc':
            # ECC parameters following BoxMOT pattern
            self.ecc_scale = float(ecc_scale)
            self.warp_mode = int(ecc_warp_mode)
            self.criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                int(ecc_max_iterations),
                float(ecc_eps)
            )

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

    def _preprocess_ecc(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for ECC (following BoxMOT pattern).

        Parameters
        ----------
        img : np.ndarray
            BGR input image

        Returns
        -------
        processed : np.ndarray
            Grayscale and scaled image ready for ECC
        """
        # Convert to grayscale if needed
        if self.grayscale and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Scale image
        if self.ecc_scale < 1.0:
            h, w = img.shape[:2]
            new_w = int(w * self.ecc_scale)
            new_h = int(h * self.ecc_scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        return img

    def applyEcc(self, raw_frame: np.ndarray, detections: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply ECC-based motion compensation (BoxMOT pattern).

        Uses OpenCV's findTransformECC for frame-to-frame motion estimation.
        Produces 2x3 affine-like matrix for TRANSLATION/EUCLIDEAN/AFFINE,
        or 3x3 homography matrix for HOMOGRAPHY mode.

        Parameters
        ----------
        raw_frame : np.ndarray
            BGR input frame
        detections : np.ndarray, optional
            Detection boxes (not used in ECC, kept for API consistency)

        Returns
        -------
        warp_matrix : np.ndarray
            2x3 or 3x3 transformation matrix
        """
        # Initialize warp matrix based on warp mode
        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Preprocess current frame
        curr = self._preprocess_ecc(raw_frame)

        # Handle first frame
        if not self.initializedFirstFrame:
            self.prevFrame = curr.copy()
            self.initializedFirstFrame = True
            return warp_matrix

        # Run ECC algorithm
        try:
            _, warp_matrix = cv2.findTransformECC(
                self.prevFrame,
                curr,
                warp_matrix,
                self.warp_mode,
                self.criteria,
                None,  # inputMask
                1      # gaussFiltSize
            )
        except cv2.error as e:
            # Handle non-convergence gracefully (BoxMOT pattern)
            try:
                if e.code == cv2.Error.StsNoConv:
                    # ECC did not converge - return identity and update prevFrame
                    self.prevFrame = curr.copy()
                    return warp_matrix
            except AttributeError:
                pass
            # For other errors, also return identity
            self.prevFrame = curr.copy()
            return warp_matrix

        # Scale translation components back to original image size
        if self.ecc_scale < 1.0:
            warp_matrix = warp_matrix.copy()
            warp_matrix[0, 2] /= self.ecc_scale
            warp_matrix[1, 2] /= self.ecc_scale

        # Update previous frame
        self.prevFrame = curr.copy()
        return warp_matrix

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
