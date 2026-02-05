# BananaTracker: Maximum Optimization Plan

## Goals
1. **Speed up** - Target 2-3x improvement (5-15 FPS → 15-30+ FPS)
2. **Better detection** - Catch more objects
3. **Solve occlusion** - Track through long occlusions
4. **Stable tracking** - Minimize ID switches

---

## Test Configuration
- **Video:** `/Users/harry/final/sample.mp4`
- **Model:** `/Users/harry/final/*.pt`
- **Masks:** ENABLED (SAM2.1 + Cutie)
- **Duration:** First 5 seconds only

---

## Part 1: Performance Optimizations (Speed)

### 1.1 Batch SAM2.1 Inference (CRITICAL)
**File:** `bananatracker/mask_propagation/mask_manager.py:222-255`

**Problem:** Processes each box individually in a loop (N boxes = N forward passes)

**Fix:** Batch all boxes in single forward pass:
```python
def _sam2_predict_boxes(self, image: np.ndarray, boxes_xyxy: List[List[float]]) -> np.ndarray:
    H, W = image.shape[:2]
    if len(boxes_xyxy) == 0:
        return np.zeros((0, H, W), dtype=np.uint8)
    if self.sam2_model is None:
        return self._bbox_to_mask_fallback(image, boxes_xyxy)

    try:
        # BATCH ALL BOXES IN ONE CALL
        input_boxes = [[[box[0], box[1], box[2], box[3]]] for box in boxes_xyxy]

        inputs = self.sam2_processor(
            images=[image] * len(boxes_xyxy),
            input_boxes=input_boxes,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.sam2_model(**inputs, multimask_output=False)

        pred_masks = outputs.pred_masks  # (N, 1, 1, H', W')
        pred_masks = F.interpolate(
            pred_masks.squeeze(2).float(),  # (N, 1, H', W')
            size=(H, W),
            mode='bilinear',
            align_corners=False
        )
        masks = (pred_masks.squeeze(1) > 0.5).cpu().numpy().astype(np.uint8)
        return masks
    except Exception as e:
        print(f"Batch SAM2 failed: {e}, falling back to sequential")
        return self._sam2_predict_boxes_sequential(image, boxes_xyxy)
```

### 1.2 Pre-compute Mask Statistics (CRITICAL)
**File:** `bananatracker/tracker/banana_tracker.py:264-362`

**Problem:** `np.unique()` called inside nested O(T*D) loop

**Fix:** Pre-compute everything before the loop:
```python
def conditioned_assignment(self, dists, max_cost, strack_pool, detections,
                           prediction_mask, tracklet_mask_dict, mask_avg_prob_dict, img_info):
    dists_cp = np.copy(dists)

    if prediction_mask is None or tracklet_mask_dict is None or len(tracklet_mask_dict) == 0:
        matches, u_track, u_detection = matching.linear_assignment(dists_cp, thresh=max_cost)
        return matches, u_track, u_detection, dists_cp

    # PRE-COMPUTE ALL MASK STATISTICS ONCE
    visible_mask_ids = set(np.unique(prediction_mask)) - {0}
    mask_stats = {}
    for mask_id in visible_mask_ids:
        if mask_avg_prob_dict.get(mask_id, 0) >= MIN_MASK_AVG_CONF:
            mask_pixels = (prediction_mask == mask_id)
            mask_stats[mask_id] = {
                'total': mask_pixels.sum(),
                'pixels': mask_pixels
            }

    img_h, img_w = img_info[0], img_info[1]

    # Build track-to-mask mapping once
    track_mask_map = {}
    for i, strack in enumerate(strack_pool):
        strack_id = strack.track_id
        if strack_id in tracklet_mask_dict:
            mask_id = tracklet_mask_dict[strack_id]
            if mask_id in mask_stats:
                track_mask_map[i] = (mask_id, mask_stats[mask_id])

    # Process only tracks with valid masks
    for i in range(dists_cp.shape[0]):
        if i not in track_mask_map:
            continue
        mask_id, stats = track_mask_map[i]
        mask_total = stats['total']
        mask_pixels = stats['pixels']

        for j in range(dists_cp.shape[1]):
            if dists[i, j] > max_cost:
                continue

            # Check for ambiguous matches
            if not (np.sum(dists[i, :] <= max_cost) > 1 or np.sum(dists[:, j] <= max_cost) > 1):
                dists_cp[i, :] += 10
                dists_cp[:, j] += 10
                dists_cp[i, j] = dists[i, j]
            else:
                det = detections[j]
                x, y, w, h = det.tlwh
                x, y = max(0, int(x)), max(0, int(y))
                hor_bound = min(img_w, x + int(w))
                ver_bound = min(img_h, y + int(h))
                box_area = (ver_bound - y) * (hor_bound - x)

                if box_area > 0 and mask_total > 0:
                    mask_in_box = mask_pixels[y:ver_bound, x:hor_bound].sum()
                    mc = mask_in_box / mask_total
                    mf = mask_in_box / box_area

                    if mf >= MIN_MM2 and mc >= MIN_MM1:
                        dists_cp[i, j] -= mf

    matches, u_track, u_detection = matching.linear_assignment(dists_cp, thresh=max_cost)
    return matches, u_track, u_detection, dists_cp
```

### 1.3 Cache IoU Distance Computation
**File:** `bananatracker/tracker/banana_tracker.py:468-523`

**Problem:** IoU computed 3 times per frame (lines 469, 498, 523)

**Fix:** Compute once and slice:
```python
# In update() after Kalman prediction (line 457):
# Create combined detections list for single IoU computation
all_dets_for_iou = detections + (detections_second if detections_second else [])

if len(strack_pool) > 0 and len(all_dets_for_iou) > 0:
    full_iou_matrix = matching.iou_distance(strack_pool, all_dets_for_iou)
else:
    full_iou_matrix = np.empty((len(strack_pool), len(all_dets_for_iou)))

# First association: slice for high-conf detections
n_high = len(detections)
dists = full_iou_matrix[:, :n_high] if n_high > 0 else np.empty((len(strack_pool), 0))
```

### 1.4 Eliminate Redundant Frame Copies
**Files:** `bananatracker/pipeline.py`, `bananatracker/visualizer.py`

**Problem:** Frame copied 2-3 times per frame (~18-27MB at 1080p)

**Fix:** Add copy parameter, copy only once:
```python
# visualizer.py - modify draw_tracks signature
def draw_tracks(self, frame: np.ndarray, tracks: List, copy: bool = True) -> np.ndarray:
    if copy:
        frame = frame.copy()
    # ... rest of method

# pipeline.py - when visualizer already copies, don't copy again
vis_frame = self.visualizer.draw_tracks(frame, tracks, copy=True)  # Single copy
```

### 1.5 Vectorize Kalman Multi-Predict
**File:** `bananatracker/tracker/banana_tracker.py:70-82`

**Problem:** Loop for setting velocity to 0 for lost tracks

**Fix:** Vectorized operation:
```python
@staticmethod
def multi_predict(stracks):
    if len(stracks) > 0:
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])

        # VECTORIZED: Set velocity to 0 for non-tracked states
        states = np.array([st.state for st in stracks])
        non_tracked_mask = states != TrackState.Tracked
        multi_mean[non_tracked_mask, 6:8] = 0  # Zero velocity for lost tracks

        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov
```

---

## Part 2: Tracking Reliability (Less ID Switches)

### 2.1 Lower New Track Creation Threshold
**File:** `bananatracker/tracker/banana_tracker.py:247`

**Problem:** `det_thresh = track_thresh + 0.1 = 0.7` - too high

**Fix:**
```python
# FROM:
self.det_thresh = track_thresh + 0.1

# TO:
self.det_thresh = track_thresh  # Same as track_thresh (0.6)
```

### 2.2 Add Grace Period for Unconfirmed Tracks
**File:** `bananatracker/tracker/banana_tracker.py`

**Problem:** Unconfirmed tracks removed after just 1 frame (line 537-540)

**Fix:** Add `unconfirmed_frames` counter to STrack:
```python
# In STrack.__init__ (add after line 56):
self.unconfirmed_frames = 0

# In update() Step 4 (replace lines 537-540):
for it in u_unconfirmed:
    track = unconfirmed[it]
    track.unconfirmed_frames += 1
    if track.unconfirmed_frames > 3:  # 4-frame grace period
        track.mark_removed()
        removed_stracks.append(track)
    # else: keep in unconfirmed pool (don't add to removed)
```

### 2.3 Use Buffered IoU for Lost Track Recovery
**File:** `bananatracker/tracker/banana_tracker.py:453-476`

**Problem:** Lost tracks match with regular IoU - often fails after occlusion

**Fix:** Use buffered (expanded) bounding boxes for lost tracks:
```python
# After strack_pool creation (line 454), separate lost and tracked:
lost_tracks = [t for t in strack_pool if t.state == TrackState.Lost]
tracked_tracks = [t for t in strack_pool if t.state == TrackState.Tracked]

# Compute IoU differently for lost vs tracked
if len(tracked_tracks) > 0 and len(detections) > 0:
    dists_tracked = matching.iou_distance(tracked_tracks, detections)
else:
    dists_tracked = np.empty((0, len(detections)))

if len(lost_tracks) > 0 and len(detections) > 0:
    # Use BUFFERED IoU for lost tracks (30% expansion)
    dists_lost = matching.buffered_iou_distance(lost_tracks, detections, buffer_scale=0.3)
else:
    dists_lost = np.empty((0, len(detections)))

# Combine cost matrices
dists = np.vstack([dists_tracked, dists_lost]) if len(dists_tracked) > 0 or len(dists_lost) > 0 else np.empty((0, len(detections)))
```

### 2.4 Add Motion Gating for Better Matching
**File:** `bananatracker/tracker/banana_tracker.py:469-470`

**Problem:** Only IoU + score fusion, no motion gating

**Fix:** Add Kalman-based gating to reject impossible matches:
```python
# After IoU distance computation (line 469):
dists = matching.iou_distance(strack_pool, detections)

# ADD: Gate based on Kalman prediction uncertainty
dists = matching.gate_cost_matrix(
    self.kalman_filter, dists, strack_pool, detections, only_position=False
)

dists = matching.fuse_score(dists, detections)
```

### 2.5 Increase Association Cost Thresholds
**File:** `bananatracker/tracker/banana_tracker.py:26-28`

**Problem:** Cost thresholds may be too strict

**Fix:** Slightly relax thresholds:
```python
# FROM:
MAX_COST_1ST_ASSOC_STEP = 0.9
MAX_COST_2ND_ASSOC_STEP = 0.5
MAX_COST_UNCONFIRMED_ASSOC_STEP = 0.7

# TO:
MAX_COST_1ST_ASSOC_STEP = 0.95    # More lenient first match
MAX_COST_2ND_ASSOC_STEP = 0.6     # More lenient low-conf match
MAX_COST_UNCONFIRMED_ASSOC_STEP = 0.8  # More lenient unconfirmed
```

---

## Part 3: Occlusion Handling

### 3.1 Increase Track Buffer Significantly
**File:** `bananatracker/config.py:56`

**Problem:** `track_buffer=30` frames (~1 sec) - too short for long occlusions

**Fix:**
```python
# FROM:
track_buffer: int = 30

# TO:
track_buffer: int = 90  # 3 seconds at 30fps
```

### 3.2 Add New Config Parameter for Lost Track Buffer Scale
**File:** `bananatracker/config.py`

**Add:**
```python
# Lost track recovery
lost_track_buffer_scale: float = 0.3  # Expand bbox by 30% for lost track matching
```

---

## Part 4: Detection Improvements

### 4.1 Lower Detection Confidence Threshold
**File:** `bananatracker/config.py:47`

**Problem:** `detection_conf_thresh=0.5` - may miss faint objects

**Fix:**
```python
# FROM:
detection_conf_thresh: float = 0.5

# TO:
detection_conf_thresh: float = 0.4  # Catch more objects
```

### 4.2 Lower Tracking Threshold
**File:** `bananatracker/config.py:55`

**Fix:**
```python
# FROM:
track_thresh: float = 0.6

# TO:
track_thresh: float = 0.5  # Match more detections in first pass
```

---

## Part 5: Debug Logging

### 5.1 Add Track Loss Logger
**File:** `bananatracker/tracker/banana_tracker.py`

**Add at top:**
```python
import logging
logger = logging.getLogger("BananaTracker")
```

**Add logging in update():**
```python
# When marking lost (line 518):
logger.debug(f"Track {track.track_id} LOST: frame={self.frame_id}, last_conf={track.score:.2f}")

# When removing (line 539):
logger.debug(f"Track {track.track_id} REMOVED: unconfirmed_frames={track.unconfirmed_frames}")

# When creating new (line 547):
logger.debug(f"Track {track.track_id} CREATED: conf={track.score:.2f}")

# When rejecting (line 545):
logger.debug(f"Detection REJECTED: conf={track.score:.2f} < thresh={self.det_thresh:.2f}")
```

### 5.2 Add Debug Config
**File:** `bananatracker/config.py`

**Add:**
```python
# Debug
debug_tracking: bool = False
```

---

## Files to Modify (Priority Order)

| Priority | File | Changes |
|----------|------|---------|
| 1 | `bananatracker/mask_propagation/mask_manager.py` | Batch SAM2.1 inference |
| 2 | `bananatracker/tracker/banana_tracker.py` | Pre-compute masks, cache IoU, buffered lost-track matching, motion gating, grace period, logging |
| 3 | `bananatracker/config.py` | Lower thresholds, increase buffer, add debug options |
| 4 | `bananatracker/pipeline.py` | Eliminate frame copies |
| 5 | `bananatracker/visualizer.py` | Add copy parameter |

---

## Summary of Changes

| Category | Change | Impact |
|----------|--------|--------|
| **Speed** | Batch SAM2.1 | 2-4x faster mask generation |
| **Speed** | Pre-compute mask stats | 5-10x faster conditioned_assignment |
| **Speed** | Cache IoU | 3x less computation |
| **Speed** | Eliminate frame copies | 18-27MB/frame memory saved |
| **Reliability** | Lower det_thresh 0.7→0.6 | More tracks created |
| **Reliability** | Grace period 1→4 frames | Stabilize new tracks |
| **Reliability** | Buffered IoU for lost | Better re-association |
| **Reliability** | Motion gating | Reject impossible matches |
| **Occlusion** | track_buffer 30→90 | 3-second recovery window |
| **Detection** | conf_thresh 0.5→0.4 | Catch more objects |
| **Detection** | track_thresh 0.6→0.5 | Match more in first pass |

---

## Verification

1. **Run baseline test:**
```bash
python test_tracking.py --video /Users/harry/final/sample.mp4 --weights /Users/harry/final/*.pt --seconds 5
```

2. **Metrics to compare:**
   - FPS (target: 15-30+)
   - Unique track IDs (fewer = better stability)
   - Track continuity through occlusions

3. **Debug logging:**
   - Enable `debug_tracking=True`
   - Check logs for LOST/REMOVED/REJECTED events
