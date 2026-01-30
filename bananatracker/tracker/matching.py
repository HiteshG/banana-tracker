"""Matching functions for track association.

Ported from McByte with fixed imports.
"""

import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from . import kalman_filter


def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    """Solve the linear assignment problem using the Hungarian algorithm.

    Parameters
    ----------
    cost_matrix : ndarray
        Cost matrix of shape (N, M).
    thresh : float
        Maximum cost threshold for valid assignments.

    Returns
    -------
    matches : ndarray
        Array of matched indices (N_matches, 2).
    unmatched_a : ndarray
        Indices of unmatched rows.
    unmatched_b : ndarray
        Indices of unmatched columns.
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """Compute IoU between two sets of bounding boxes.

    Parameters
    ----------
    atlbrs : list[tlbr] | np.ndarray
        First set of boxes in tlbr format.
    btlbrs : list[tlbr] | np.ndarray
        Second set of boxes in tlbr format.

    Returns
    -------
    ious : np.ndarray
        IoU matrix of shape (len(atlbrs), len(btlbrs)).
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float64)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float64),
        np.ascontiguousarray(btlbrs, dtype=np.float64)
    )

    return ious


def tlbr_expand(tlbr, scale=1.2):
    """Expand a bounding box by a scale factor."""
    w = tlbr[2] - tlbr[0]
    h = tlbr[3] - tlbr[1]

    half_scale = 0.5 * scale

    tlbr[0] -= half_scale * w
    tlbr[1] -= half_scale * h
    tlbr[2] += half_scale * w
    tlbr[3] += half_scale * h

    return tlbr


def iou_distance(atracks, btracks):
    """Compute cost based on IoU.

    Parameters
    ----------
    atracks : list[STrack]
        First set of tracks.
    btracks : list[STrack]
        Second set of tracks.

    Returns
    -------
    cost_matrix : np.ndarray
        Cost matrix where cost = 1 - IoU.
    """
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def buffered_iou_distance(atracks, btracks, buffer_scale):
    """Compute cost based on buffered IoU (expanded bounding boxes)."""
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        raise Exception("Unimplemented handling for buffered IOU")
    else:
        atlbrs = get_buffered_tlbrs(atracks, buffer_scale)
        btlbrs = get_buffered_tlbrs(btracks, buffer_scale)
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def get_buffered_tlbrs(tracks, buffer_scale):
    """Get buffered tlbr coordinates for a list of tracks."""
    tlbrs = []
    for t in tracks:
        t_left, t_top, t_right, t_bottom = t.tlbr

        b_tlbr = np.asarray(t.tlbr).copy()
        b_tlbr[0] = t_left - buffer_scale * (t_right - t_left)
        b_tlbr[1] = t_top - buffer_scale * (t_bottom - t_top)
        b_tlbr[2] = t_right + buffer_scale * (t_right - t_left)
        b_tlbr[3] = t_bottom + buffer_scale * (t_bottom - t_top)

        tlbrs.append(b_tlbr)

    return tlbrs


def v_iou_distance(atracks, btracks):
    """Compute cost based on IoU using predicted bboxes."""
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """Compute embedding distance between tracks and detections.

    Parameters
    ----------
    tracks : list[STrack]
        List of tracks.
    detections : list[BaseTrack]
        List of detections.
    metric : str
        Distance metric ('cosine', 'euclidean', etc.).

    Returns
    -------
    cost_matrix : np.ndarray
        Cost matrix based on embedding distance.
    """
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float64)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float64)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    """Gate cost matrix using Mahalanobis distance."""
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    """Fuse motion information into cost matrix."""
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    """Fuse IoU similarity with appearance similarity."""
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    """Fuse detection scores into cost matrix.

    Parameters
    ----------
    cost_matrix : np.ndarray
        IoU-based cost matrix.
    detections : list
        List of detections with .score attribute.

    Returns
    -------
    fuse_cost : np.ndarray
        Score-fused cost matrix.
    """
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost
