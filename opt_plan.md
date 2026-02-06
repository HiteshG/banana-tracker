# BananaTracker: Maximum Optimization Plan

## Goals
2. **Better detection** - Catch more objects
3. **Solve occlusion** - Track through long occlusions
4. **Stable tracking** - Minimize ID switches

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

| **Reliability** | Lower det_thresh 0.7→0.6 | More tracks created |
| **Reliability** | Grace period 1→4 frames | Stabilize new tracks |
| **Reliability** | Buffered IoU for lost | Better re-association |
| **Reliability** | Motion gating | Reject impossible matches |
| **Occlusion** | track_buffer 30→90 | 3-second recovery window |
| **Detection** | conf_thresh 0.5→0.4 | Catch more objects |
| **Detection** | track_thresh 0.6→0.5 | Match more in first pass |