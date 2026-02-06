"""BananaTracker: ByteTrack-based multi-object tracker.

Adapted from McByte's McByteTracker with the following changes:
- Renamed McByteTracker → BananaTracker
- Added class_id attribute to STrack for per-class visualization
- Simplified update method (removed YOLOX-specific preprocessing)
- Removed logging (can be added back if needed)
- Kept conditioned_assignment for future mask integration
"""

import numpy as np
from collections import deque
import copy

from .kalman_filter import KalmanFilter
from . import matching
from .gmc import GMC
from .basetrack import BaseTrack, TrackState


# Constants for association steps
MIN_MASK_AVG_CONF = 0.6
MIN_MM1 = 0.9  # mc threshold
MIN_MM2 = 0.05  # mf threshold

MAX_COST_1ST_ASSOC_STEP = 0.95    # More lenient first match
MAX_COST_2ND_ASSOC_STEP = 0.6     # More lenient low-conf match
MAX_COST_UNCONFIRMED_ASSOC_STEP = 0.8  # More lenient unconfirmed


class STrack(BaseTrack):
    """Single object track class with Kalman filter motion model."""

    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, class_id=-1):
        """Initialize a track.

        Parameters
        ----------
        tlwh : array-like
            Bounding box in (top, left, width, height) format.
        score : float
            Detection confidence score.
        class_id : int
            Class ID for the detection (-1 if unknown).
        """
        # Wait for activation
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.class_id = class_id
        self.unconfirmed_frames = 0  # Grace period counter for unconfirmed tracks

        # Store last detection for visualization
        self.last_det_tlwh = tlwh

    def predict(self):
        """Predict the next state using Kalman filter."""
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """Predict next states for multiple tracks (vectorized)."""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """Apply camera motion compensation to multiple tracks."""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """Re-activate a lost track."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh)
        )

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.class_id = new_track.class_id
        self.last_det_tlwh = new_track.tlwh

    def update(self, new_track, frame_id):
        """Update a matched track with new detection.

        Parameters
        ----------
        new_track : STrack
            New detection to update with.
        frame_id : int
            Current frame number.
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh)
        )

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.class_id = new_track.class_id
        self.last_det_tlwh = new_track.tlwh

    @property
    def tlwh(self):
        """Get current position in (top, left, width, height) format."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Get current position in (x1, y1, x2, y2) format."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Get current position in (center_x, center_y, width, height) format."""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert (top, left, width, height) to (center_x, center_y, aspect_ratio, height)."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert (top, left, width, height) to (center_x, center_y, width, height)."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        """Get current position in (center_x, center_y, width, height) format."""
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """Convert (x1, y1, x2, y2) to (top, left, width, height)."""
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        """Convert (top, left, width, height) to (x1, y1, x2, y2)."""
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return f'OT_{self.track_id}_({self.start_frame}-{self.end_frame})'


class BananaTracker:
    """ByteTrack-based multi-object tracker.

    Based on McByte's McByteTracker with conditioned_assignment for future
    mask integration support.
    """

    def __init__(self, track_thresh=0.6, track_buffer=30, match_thresh=0.8,
                 frame_rate=30, cmc_method='orb'):
        """Initialize the tracker.

        Parameters
        ----------
        track_thresh : float
            Detection confidence threshold for first association.
        track_buffer : int
            Number of frames to keep lost tracks.
        match_thresh : float
            Matching threshold for association.
        frame_rate : int
            Video frame rate.
        cmc_method : str
            Camera motion compensation method.
        """
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.track_thresh = track_thresh
        self.det_thresh = track_thresh  # Same as track_thresh for more track creation
        # Cap buffer at 45 frames max to avoid memory issues
        self.buffer_size = min(int(frame_rate / 30.0 * track_buffer), 45)
        self.max_time_lost = self.buffer_size
        self.match_thresh = match_thresh
        self.kalman_filter = KalmanFilter()

        # Camera motion compensation
        self.gmc = GMC(method=cmc_method, verbose=None)

    def reset(self):
        """Reset the tracker state."""
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        BaseTrack.reset_id()

    def conditioned_assignment(self, dists, max_cost, strack_pool, detections,
                               prediction_mask, tracklet_mask_dict, mask_avg_prob_dict, img_info):
        """Perform assignment with optional mask-based cost enrichment.

        When mask parameters are None or empty, behaves as standard ByteTrack.
        When masks are provided, enriches cost matrix with mask metrics (mc, mf).

        Parameters
        ----------
        dists : np.ndarray
            IoU-based cost matrix.
        max_cost : float
            Maximum cost threshold.
        strack_pool : list[STrack]
            Pool of tracks.
        detections : list[STrack]
            Current detections.
        prediction_mask : np.ndarray or None
            Propagated mask tensor (H, W) with unique values per object.
        tracklet_mask_dict : dict or None
            Mapping of track_id → mask color value.
        mask_avg_prob_dict : dict or None
            Mapping of mask_id → confidence (0-1).
        img_info : tuple
            Image dimensions (height, width).

        Returns
        -------
        matches : np.ndarray
            Matched pairs (track_idx, det_idx).
        u_track : np.ndarray
            Unmatched track indices.
        u_detection : np.ndarray
            Unmatched detection indices.
        dists_cp : np.ndarray
            Modified cost matrix.
        """
        dists_cp = np.copy(dists)

        # Skip mask processing if no masks provided
        if prediction_mask is None or tracklet_mask_dict is None or mask_avg_prob_dict is None:
            matches, u_track, u_detection = matching.linear_assignment(dists_cp, thresh=max_cost)
            return matches, u_track, u_detection, dists_cp

        if len(tracklet_mask_dict) == 0:
            matches, u_track, u_detection = matching.linear_assignment(dists_cp, thresh=max_cost)
            return matches, u_track, u_detection, dists_cp

        # Process each entry in the cost matrix
        for i in range(dists_cp.shape[0]):
            for j in range(dists_cp.shape[1]):
                if dists[i, j] <= max_cost:
                    # Check if there are other entries meeting the threshold
                    if not (sum(dists[i, :] <= max_cost) > 1 or sum(dists[:, j] <= max_cost) > 1):
                        # Clear match - set all others to high cost
                        dists_cp[i, :] += 10
                        dists_cp[:, j] += 10
                        dists_cp[i, j] = dists[i, j]
                    else:
                        # Ambiguous match - use mask cue if available
                        strack = strack_pool[i]
                        det = detections[j]

                        strack_id = strack.track_id
                        if strack_id in tracklet_mask_dict:
                            strack_mask_id = tracklet_mask_dict[strack_id]

                            # Check if mask is visible on scene
                            if strack_mask_id in list(np.unique(prediction_mask))[1:]:
                                # Check mask confidence
                                if mask_avg_prob_dict.get(strack_mask_id, 0) >= MIN_MASK_AVG_CONF:
                                    img_h, img_w = img_info[0], img_info[1]

                                    # Get detection coordinates
                                    x, y, w, h = det.tlwh
                                    x = max(0, int(x))
                                    y = max(0, int(y))
                                    w = int(w)
                                    h = int(h)
                                    hor_bound = min(img_w, x + w)
                                    ver_bound = min(img_h, y + h)

                                    # Compute mc and mf
                                    mask_in_box = (prediction_mask[y:ver_bound, x:hor_bound] == strack_mask_id).sum()
                                    mask_total = (prediction_mask == strack_mask_id).sum()
                                    box_area = (ver_bound - y) * (hor_bound - x)

                                    if mask_total > 0 and box_area > 0:
                                        mask_match_opt_1 = mask_in_box / mask_total  # mc
                                        mask_match_opt_2 = mask_in_box / box_area  # mf

                                        # Apply mask cue if conditions met
                                        if mask_match_opt_2 >= MIN_MM2 and mask_match_opt_1 >= MIN_MM1:
                                            dists_cp[i, j] -= mask_match_opt_2

        # Perform Hungarian matching
        matches, u_track, u_detection = matching.linear_assignment(dists_cp, thresh=max_cost)

        return matches, u_track, u_detection, dists_cp

    def update(self, detections_array, img_info, img_size=None,
               prediction_mask=None, tracklet_mask_dict=None, mask_avg_prob_dict=None,
               frame_img=None):
        """Update tracks with new detections.

        Parameters
        ----------
        detections_array : np.ndarray
            Detection array of shape (N, 5) or (N, 6):
            - (N, 5): [x1, y1, x2, y2, conf]
            - (N, 6): [x1, y1, x2, y2, conf, class_id]
        img_info : tuple
            Image dimensions (height, width).
        img_size : tuple, optional
            Model input size for scaling (not used if detections already scaled).
        prediction_mask : np.ndarray, optional
            Propagated mask tensor for mask-based association.
        tracklet_mask_dict : dict, optional
            Mapping of track_id → mask color.
        mask_avg_prob_dict : dict, optional
            Mapping of mask_id → confidence.
        frame_img : np.ndarray, optional
            BGR frame for camera motion compensation.

        Returns
        -------
        output_stracks : list[STrack]
            Currently active tracks.
        removed_track_ids : list[int]
            IDs of tracks removed this frame.
        new_tracks : list[STrack]
            Newly confirmed tracks this frame.
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Parse detections
        if detections_array is None or len(detections_array) == 0:
            detections_array = np.empty((0, 5))

        if detections_array.shape[1] == 5:
            scores = detections_array[:, 4]
            bboxes = detections_array[:, :4]
            class_ids = np.full(len(scores), -1, dtype=np.int32)
        elif detections_array.shape[1] >= 6:
            scores = detections_array[:, 4]
            bboxes = detections_array[:, :4]
            class_ids = detections_array[:, 5].astype(np.int32)
        else:
            scores = np.array([])
            bboxes = np.empty((0, 4))
            class_ids = np.array([], dtype=np.int32)

        img_h, img_w = img_info[0], img_info[1]

        # Split detections by confidence
        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        class_ids_keep = class_ids[remain_inds]
        class_ids_second = class_ids[inds_second]

        # Create STrack detections
        if len(dets) > 0:
            detections = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, cls_id)
                for tlbr, s, cls_id in zip(dets, scores_keep, class_ids_keep)
            ]
        else:
            detections = []

        # Step 1: Separate confirmed and unconfirmed tracks
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # Step 2: First association with high score detections
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict current locations with Kalman filter
        STrack.multi_predict(strack_pool)

        # Camera motion compensation
        if frame_img is not None:
            try:
                warp = self.gmc.apply(frame_img, dets)
                STrack.multi_gmc(strack_pool, warp)
                STrack.multi_gmc(unconfirmed, warp)
            except Exception:
                pass

        # Separate lost and tracked for different IoU treatment
        lost_tracks = [t for t in strack_pool if t.state == TrackState.Lost]
        active_tracks = [t for t in strack_pool if t.state == TrackState.Tracked]

        # Compute IoU distance (use buffered IoU for lost tracks)
        if len(active_tracks) > 0 and len(detections) > 0:
            dists_active = matching.iou_distance(active_tracks, detections)
        else:
            dists_active = np.empty((0, len(detections)))

        if len(lost_tracks) > 0 and len(detections) > 0:
            # Use buffered IoU for lost tracks (30% expansion for better re-association)
            dists_lost = matching.buffered_iou_distance(lost_tracks, detections, buffer_scale=0.3)
        else:
            dists_lost = np.empty((0, len(detections)))

        # Combine cost matrices and reconstruct strack_pool order
        combined_tracks = active_tracks + lost_tracks
        if len(dists_active) > 0 or len(dists_lost) > 0:
            dists = np.vstack([dists_active, dists_lost]) if len(dists_active) > 0 and len(dists_lost) > 0 else (dists_active if len(dists_active) > 0 else dists_lost)
        else:
            dists = np.empty((0, len(detections)))

        # Update strack_pool to match combined order
        strack_pool = combined_tracks

        # Add motion gating to reject impossible matches
        dists = matching.gate_cost_matrix(
            self.kalman_filter, dists, strack_pool, detections, only_position=False
        )

        dists = matching.fuse_score(dists, detections)

        # Conditioned assignment (with mask support)
        matches, u_track, u_detection, _ = self.conditioned_assignment(
            dists, MAX_COST_1ST_ASSOC_STEP, strack_pool, detections,
            prediction_mask, tracklet_mask_dict, mask_avg_prob_dict, img_info
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # Step 3: Second association with low score detections
        if len(dets_second) > 0:
            detections_second = [
                STrack(STrack.tlbr_to_tlwh(tlbr), s, cls_id)
                for tlbr, s, cls_id in zip(dets_second, scores_second, class_ids_second)
            ]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)

        matches, u_track, u_detection_second, _ = self.conditioned_assignment(
            dists, MAX_COST_2ND_ASSOC_STEP, r_tracked_stracks, detections_second,
            prediction_mask, tracklet_mask_dict, mask_avg_prob_dict, img_info
        )

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # Step 4: Deal with unconfirmed tracks
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        dists = matching.fuse_score(dists, detections)

        matches, u_unconfirmed, u_detection, _ = self.conditioned_assignment(
            dists, MAX_COST_UNCONFIRMED_ASSOC_STEP, unconfirmed, detections,
            prediction_mask, tracklet_mask_dict, mask_avg_prob_dict, img_info
        )

        new_confirmed_tracks = []
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            new_confirmed_tracks.append(unconfirmed[itracked])
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.unconfirmed_frames += 1
            if track.unconfirmed_frames > 3:  # 4-frame grace period
                track.mark_removed()
                removed_stracks.append(track)
            # else: keep in unconfirmed pool for next frame

        # Step 5: Initialize new tracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        # Step 6: Update state
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        # Output currently tracked objects
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        removed_track_ids = [track.track_id for track in removed_stracks]

        return output_stracks, removed_track_ids, new_confirmed_tracks


def joint_stracks(tlista, tlistb):
    """Join two track lists without duplicates."""
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    """Subtract tlistb from tlista."""
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    """Remove duplicate tracks based on IoU."""
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
