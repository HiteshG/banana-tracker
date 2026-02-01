# Plan: Replace SAM + Cutie with SAM2.1 in McByte MaskManager

## Summary

Replace the current two-component mask propagation system (SAM1 + Cutie) with SAM2.1's unified video object segmentation pipeline using the `segment-anything-2-real-time` fork. SAM2.1 has built-in memory management, eliminating redundancy and simplifying architecture.

---

## Architecture Change

**Current (SAM1 + Cutie):**
```
SAM1: Creates initial masks from bboxes
Cutie: Propagates masks temporally with memory
       ├─ InferenceCore.step(frame, masks) - init memory
       ├─ InferenceCore.step(frame) - propagate
       └─ object_manager.purge_selected_objects() - remove
```

**Target (SAM2.1 unified):**
```
SAM2.1 Camera Predictor:
       ├─ load_first_frame(frame) - init
       ├─ add_new_prompt(frame_idx, obj_id, bbox) - create mask + add to memory
       ├─ track(frame) - propagate all objects
       └─ Soft removal (no direct delete API)
```

---

## Key API Mappings

| Operation | Current | SAM2.1 |
|-----------|---------|--------|
| Init model | `CUTIE(cfg)` + `InferenceCore()` | `build_sam2_camera_predictor(config, ckpt)` |
| Load frame | `processor.step(frame, mask)` | `predictor.load_first_frame(frame)` |
| Create mask | `SamPredictor.predict_torch(boxes=...)` | `predictor.add_new_prompt(frame_idx, obj_id, bbox)` |
| Propagate | `processor.step(frame)` | `predictor.track(frame)` |
| Add object | `processor.step(frame, extended_mask)` | `predictor.add_new_prompt(frame_idx, obj_id, bbox)` |
| Remove object | `purge_selected_objects()` | Soft removal (filter in post-processing) |

**Bbox format change:**
- Current: `[x1, y1, x2, y2]`
- SAM2.1: `[[x1, y1], [x2, y2]]` (2x2 numpy array)

**Output format:**
- Current: Probability tensor `(num_objects+1, H, W)`
- SAM2.1: Logits tensor `(num_objects, 1, H, W)` - threshold at 0.0

---

## Implementation Steps

### Step 1: Update Imports in `mask_manager.py`

**Remove:**
```python
from segment_anything import sam_model_registry, SamPredictor
from cutie.model.cutie import CUTIE
from cutie.inference.inference_core import InferenceCore
from gui.interactive_utils import image_to_torch, torch_prob_to_numpy_mask, index_numpy_to_one_hot_torch
```

**Add:**
```python
import sys
sys.path.insert(0, '/Users/harry/final/segment-anything-2-real-time')
from sam2.build_sam import build_sam2_camera_predictor
import torch.nn.functional as F
```

### Step 2: Rewrite `__init__` Method

**New state variables:**
```python
self.predictor = None  # SAM2.1 camera predictor
self.first_frame_loaded = False
self.tracklet_to_sam2_obj = {}  # track_id -> SAM2 obj_id
self.sam2_obj_counter = 0
self.removed_obj_ids = set()  # Soft-removed objects
self.frame_counter = 0  # Track frames for SAM2.1
```

**Initialize SAM2.1:**
```python
SAM2_CONFIG = "/Users/harry/final/segment-anything-2-real-time/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT = "/Users/harry/final/segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"

self.predictor = build_sam2_camera_predictor(
    SAM2_CONFIG, SAM2_CHECKPOINT, device=self.device
)
```

### Step 3: Rewrite `initialize_first_masks` Method

**Key changes:**
1. Call `predictor.load_first_frame(frame)` first
2. For each bbox, call `predictor.add_new_prompt(frame_idx=0, obj_id, bbox)`
3. Convert bbox from `[x1,y1,x2,y2]` to `[[x1,y1],[x2,y2]]`
4. Build combined mask from individual SAM2.1 outputs

```python
def _initialize_first_masks(self, frame, online_tlwhs, online_ids):
    # Load first frame
    self.predictor.load_first_frame(frame)
    self.first_frame_loaded = True

    # Filter overlapping tracklets (keep existing logic)
    # ...

    # Add each object
    all_masks = []
    for bbox, track_id in zip(image_boxes_list, new_tracks_id):
        self.sam2_obj_counter += 1
        sam2_obj_id = self.sam2_obj_counter

        # Convert bbox: [x1,y1,w,h] -> [[x1,y1],[x2,y2]]
        bbox_sam2 = np.array([[x1, y1], [x1+w, y1+h]], dtype=np.float32)

        _, obj_ids, mask_logits = self.predictor.add_new_prompt(
            frame_idx=0, obj_id=sam2_obj_id, bbox=bbox_sam2
        )

        self.tracklet_to_sam2_obj[track_id] = sam2_obj_id
        all_masks.append((sam2_obj_id, mask_logits))

    return self._combine_masks(all_masks, frame.shape[:2])
```

### Step 4: Rewrite `add_new_masks` Method

**Key changes:**
1. Use `predictor.add_new_prompt()` with current frame_idx
2. SAM2.1 automatically handles memory when adding new prompts

```python
def _add_new_masks(self, frame, online_tlwhs, online_ids, new_tracks):
    # Filter overlapping tracklets (keep existing logic)
    # ...

    for bbox, track_id in zip(image_boxes_list, new_tracks_id):
        self.sam2_obj_counter += 1
        sam2_obj_id = self.sam2_obj_counter

        bbox_sam2 = np.array([[x1, y1], [x2, y2]], dtype=np.float32)

        # Add to SAM2.1 at current frame
        _, obj_ids, mask_logits = self.predictor.add_new_prompt(
            frame_idx=self.frame_counter,
            obj_id=sam2_obj_id,
            bbox=bbox_sam2
        )

        self.tracklet_to_sam2_obj[track_id] = sam2_obj_id
        # Update tracklet_mask_dict (keep existing logic)
```

### Step 5: Rewrite `remove_masks` Method

**Strategy: Soft removal** (SAM2.1 has no direct delete API)

```python
def _remove_masks(self, removed_tracks_ids):
    for track_id in removed_tracks_ids:
        if track_id in self.tracklet_to_sam2_obj:
            sam2_obj_id = self.tracklet_to_sam2_obj[track_id]
            self.removed_obj_ids.add(sam2_obj_id)

        # Update tracklet_mask_dict (keep existing renumbering logic)
```

**Optional: Hard reset when too many removed objects:**
```python
if len(self.removed_obj_ids) > 10:
    self._hard_reset_and_reinitialize(frame, online_tlwhs, online_ids)
```

### Step 6: Add `_propagate_masks` Method

**New method to replace `processor.step(frame)`:**

```python
def _propagate_masks(self, frame):
    # Propagate to current frame
    obj_ids, mask_logits = self.predictor.track(frame)
    self.frame_counter += 1

    # Filter out soft-removed objects
    active_masks = []
    for i, obj_id in enumerate(obj_ids):
        if obj_id not in self.removed_obj_ids:
            active_masks.append((obj_id, mask_logits[i:i+1]))

    return self._combine_masks(active_masks, frame.shape[:2])
```

### Step 7: Add `_combine_masks` Helper

**Convert SAM2.1 outputs to McByte format:**

```python
def _combine_masks(self, masks_list, image_shape):
    """
    Convert SAM2.1 per-object logits to probability tensor.

    Input: List of (sam2_obj_id, mask_logits) where logits is (1, 1, H, W)
    Output: Tensor (num_objects+1, H, W) with probabilities
    """
    H, W = image_shape
    num_objects = len(self.tracklet_mask_dict)
    prediction = torch.zeros((num_objects + 1, H, W), device=self.device)

    for sam2_obj_id, logits in masks_list:
        # Find track_id and mask_id for this SAM2 object
        track_id = None
        for tid, sid in self.tracklet_to_sam2_obj.items():
            if sid == sam2_obj_id:
                track_id = tid
                break

        if track_id is None or track_id not in self.tracklet_mask_dict:
            continue

        mask_id = self.tracklet_mask_dict[track_id]

        # Convert logits to probability
        prob = torch.sigmoid(logits.squeeze())
        if prob.shape != (H, W):
            prob = F.interpolate(prob[None, None], (H, W), mode='bilinear').squeeze()

        prediction[mask_id] = prob

    # Background = 1 - max(foreground)
    prediction[0] = 1.0 - prediction[1:].max(dim=0).values.clamp(0, 1)

    return prediction
```

### Step 8: Update `get_updated_masks` Main Method

```python
def get_updated_masks(self, img_info, img_info_prev, frame_id,
                      online_tlwhs, online_ids, new_tracks, removed_tracks_ids):
    prediction = None

    current_frame = img_info['raw_img']  # numpy BGR
    prev_frame = img_info_prev['raw_img'] if img_info_prev else None

    if frame_id == self.SAM_START_FRAME + 1 + self.init_delay_counter:
        prediction = self._initialize_first_masks(prev_frame, online_tlwhs, online_ids)

    elif frame_id > self.SAM_START_FRAME + 1 + self.init_delay_counter:
        self._add_new_masks(prev_frame, online_tlwhs, online_ids, new_tracks)
        self._remove_masks(removed_tracks_ids)
        prediction = self._propagate_masks(current_frame)

    # Post-process (keep existing logic)
    if prediction is not None:
        prediction, mask_avg_prob_dict, colors = self.post_process_mask(prediction)

    return prediction, self.tracklet_mask_dict.copy(), mask_avg_prob_dict, colors
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `/Users/harry/final/McByte/mask_propagation/mask_manager.py` | Complete rewrite of MaskManager class |
| `/Users/harry/final/McByte/tools/demo_track.py` | Update MaskManager instantiation (optional: add config params) |

---

## Interface Preservation

**Return signature UNCHANGED:**
```python
return (prediction, tracklet_mask_dict, mask_avg_prob_dict, prediction_colors_preserved)
```

Where:
- `prediction`: np.ndarray (H, W) with unique integer IDs (0=background)
- `tracklet_mask_dict`: Dict[track_id → mask_id]
- `mask_avg_prob_dict`: Dict[mask_id → confidence (0-1)]
- `prediction_colors_preserved`: np.ndarray for visualization

**Tracker integration unchanged:**
- `conditioned_assignment()` in mcbyte_tracker.py continues to work
- Cost matrix enrichment (mc, mf thresholds) unchanged

---

## Object Removal Strategy

**Decision: Soft removal** (best for tracking quality and minimal jitter)

SAM2.1 has no `delete_object()` API. Soft removal chosen because:
- **No re-init jitter:** Hard resets cause momentary mask instability
- **Better temporal consistency:** Memory of removed objects helps with occlusion
- **Smooth transitions:** No sudden quality drops

**Implementation:**
```python
self.removed_obj_ids = set()  # Track soft-removed objects

def _remove_masks(self, removed_tracks_ids):
    for track_id in removed_tracks_ids:
        if track_id in self.tracklet_to_sam2_obj:
            sam2_obj_id = self.tracklet_to_sam2_obj[track_id]
            self.removed_obj_ids.add(sam2_obj_id)
        # Update tracklet_mask_dict with renumbering
```

**Memory safeguard (optional, low priority):** If `len(removed_obj_ids) > 50` (rare), can implement periodic cleanup during scene cuts.

---

## Dependencies

SAM2.1 real-time fork available at:
`/Users/harry/final/segment-anything-2-real-time`

**Required files:**
- Config: `sam2/configs/sam2.1/sam2.1_hiera_l.yaml` (already exists)
- Checkpoint: `checkpoints/sam2.1_hiera_large.pt` (needs download)

**Download checkpoints (required before implementation):**
```bash
cd /Users/harry/final/segment-anything-2-real-time/checkpoints
bash download_ckpts.sh
```

**Model variant: Large** (confirmed)
- `sam2.1_hiera_large.pt` - Best mask quality, chosen for tracking accuracy
- Config: `sam2/configs/sam2.1/sam2.1_hiera_l.yaml`
