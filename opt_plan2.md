
---
Plan 2
BananaTracker GPU Speed Optimization Plan                                                                                                                                   
                                                                                                                                                                             
 Problem Summary                                                                                                                                                             
                                                                                                                                                                             
 - GPU utilization: only 7GB of 48GB used                                                                                                                                    
 - Single-frame processing severely underutilizes GPU                                                                                                                        
 - ECC motion compensation is slow (5000 iterations with 2x downscale)                                                                                                       
 - No batch processing for YOLO or SAM2                                                                                                                                      
                                                                                                                                                                             
 Why Current Changes Made It Slower                                                                                                                                          
                                                                                                                                                                             
 1. ECC is slower than ORB - ECC uses iterative optimization (5000 iterations!)                                                                                              
 2. Lower detection_conf (0.5→0.4) - More detections = more work for tracker/masks                                                                                           
 3. Main bottleneck unchanged - Still processing 1 frame at a time on GPU                                                                                                    
                                                                                                                                                                             
 Target Environment                                                                                                                                                          
                                                                                                                                                                             
 - Modal notebooks (similar to Colab)                                                                                                                                        
 - 48GB GPU available                                                                                                                                                        
 - Must maintain detection/tracking quality                                                                                                                                  
                                                                                                                                                                             
 ---                                                                                                                                                                         
 Phase 1: High-Impact GPU Optimizations                                                                                                                                      
                                                                                                                                                                             
 1.1 YOLO Batch Detection (HIGHEST PRIORITY)                                                                                                                                 
                                                                                                                                                                             
 File: bananatracker/detector.py                                                                                                                                             
                                                                                                                                                                             
 Add batch detection capability:                                                                                                                                             
 def detect_batch(self, frames: List[np.ndarray]) -> List[np.ndarray]:                                                                                                       
     """Batch detect on multiple frames at once."""                                                                                                                          
     results = self.model(                                                                                                                                                   
         frames,  # Pass list of frames                                                                                                                                      
         verbose=False,                                                                                                                                                      
         conf=self.config.detection_conf_thresh,                                                                                                                             
         iou=self.config.detection_iou_thresh,                                                                                                                               
         half=True  # FP16 for speed                                                                                                                                         
     )                                                                                                                                                                       
     return [self._parse_results(r) for r in results]                                                                                                                        
                                                                                                                                                                             
 File: bananatracker/config.py                                                                                                                                               
 # Add optimization settings                                                                                                                                                 
 detection_batch_size: int = 16  # Process 16 frames at once                                                                                                                 
 use_half_precision: bool = True  # FP16 inference                                                                                                                           
                                                                                                                                                                             
 Expected: 4-6x faster detection, GPU usage 7GB → 15-20GB                                                                                                                    
                                                                                                                                                                             
 1.2 Enable FP16 (Half Precision) Everywhere                                                                                                                                 
                                                                                                                                                                             
 File: bananatracker/detector.py                                                                                                                                             
 # In __init__                                                                                                                                                               
 self.model = YOLO(config.yolo_weights)                                                                                                                                      
 self.model.to(config.device)                                                                                                                                                
 if config.use_half_precision:                                                                                                                                               
     self.model.half()  # Convert to FP16                                                                                                                                    
                                                                                                                                                                             
 File: bananatracker/mask_propagation/mask_manager.py                                                                                                                        
 # SAM2 with FP16                                                                                                                                                            
 self.sam2_model = Sam2Model.from_pretrained(sam2_model_id, token=token)                                                                                                     
 self.sam2_model.to(device).half().eval()  # Add .half()                                                                                                                     
                                                                                                                                                                             
 Expected: 30-50% memory reduction, 20-30% speed boost                                                                                                                       
                                                                                                                                                                             
 1.3 Optimize ECC for Speed (Keep Best Quality)                                                                                                                              
                                                                                                                                                                             
 File: bananatracker/tracker/gmc.py (lines 51-55)                                                                                                                            
                                                                                                                                                                             
 Current (SLOW):                                                                                                                                                             
 number_of_iterations = 5000  # Way too many!                                                                                                                                
 termination_eps = 1e-6       # Very strict                                                                                                                                  
 self.downscale = 2           # Only 2x downscale                                                                                                                            
                                                                                                                                                                             
 Optimized (keep ECC quality, improve speed):                                                                                                                                
 number_of_iterations = 500   # Sufficient for convergence                                                                                                                   
 termination_eps = 1e-4       # Slightly relaxed (still accurate)                                                                                                            
 self.downscale = 4           # 4x downscale (much faster, still good)                                                                                                       
                                                                                                                                                                             
 File: bananatracker/tracker/gmc.py - Update __init__:                                                                                                                       
 def __init__(self, method='ecc', downscale=4, ecc_iterations=500, ecc_eps=1e-4):                                                                                            
     self.downscale = max(1, int(downscale))                                                                                                                                 
                                                                                                                                                                             
     if self.method == 'ecc':                                                                                                                                                
         self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,                                                                                                   
                         ecc_iterations, ecc_eps)                                                                                                                            
                                                                                                                                                                             
 Expected: 4-8x faster ECC while maintaining accuracy                                                                                                                        
                                                                                                                                                                             
 ---                                                                                                                                                                         
 Phase 2: Batch Processing Pipeline                                                                                                                                          
                                                                                                                                                                             
 2.1 Frame Buffer for Batch Processing                                                                                                                                       
                                                                                                                                                                             
 File: bananatracker/pipeline.py                                                                                                                                             
                                                                                                                                                                             
 def process_video_batched(self, video_path: str, batch_size: int = 16):                                                                                                     
     """Process video with batched detection for GPU efficiency."""                                                                                                          
     cap = cv2.VideoCapture(video_path)                                                                                                                                      
     fps = int(cap.get(cv2.CAP_PROP_FPS))                                                                                                                                    
                                                                                                                                                                             
     # Update tracker buffer                                                                                                                                                 
     self.tracker.buffer_size = min(int(fps / 30.0 * self.config.track_buffer), 45)                                                                                          
     self.tracker.max_time_lost = self.tracker.buffer_size                                                                                                                   
                                                                                                                                                                             
     frame_buffer = []                                                                                                                                                       
     frame_ids = []                                                                                                                                                          
                                                                                                                                                                             
     while True:                                                                                                                                                             
         # Fill buffer                                                                                                                                                       
         for _ in range(batch_size):                                                                                                                                         
             ret, frame = cap.read()                                                                                                                                         
             if not ret:                                                                                                                                                     
                 break                                                                                                                                                       
             frame_buffer.append(frame)                                                                                                                                      
             frame_ids.append(len(frame_ids) + 1)                                                                                                                            
                                                                                                                                                                             
         if not frame_buffer:                                                                                                                                                
             break                                                                                                                                                           
                                                                                                                                                                             
         # Batch detect all frames at once (GPU efficient!)                                                                                                                  
         all_detections = self.detector.detect_batch(frame_buffer)                                                                                                           
                                                                                                                                                                             
         # Process each frame through tracker (must be sequential)                                                                                                           
         for frame, detections, fid in zip(frame_buffer, all_detections, frame_ids):                                                                                         
             tracks, removed_ids, new_tracks = self.tracker.update(                                                                                                          
                 detections_array=detections,                                                                                                                                
                 img_info=(frame.shape[0], frame.shape[1]),                                                                                                                  
                 frame_img=frame                                                                                                                                             
             )                                                                                                                                                               
             yield fid, frame, tracks                                                                                                                                        
                                                                                                                                                                             
         frame_buffer.clear()                                                                                                                                                
         frame_ids.clear()                                                                                                                                                   
                                                                                                                                                                             
 2.2 SAM2 Batch Box Processing                                                                                                                                               
                                                                                                                                                                             
 File: bananatracker/mask_propagation/mask_manager.py                                                                                                                        
                                                                                                                                                                             
 Replace sequential loop (lines 223-255) with batch:                                                                                                                         
 def _sam2_predict_boxes(self, image, boxes_xyxy):                                                                                                                           
     if len(boxes_xyxy) == 0:                                                                                                                                                
         return []                                                                                                                                                           
                                                                                                                                                                             
     # Batch ALL boxes in single inference call                                                                                                                              
     with torch.cuda.amp.autocast(dtype=torch.float16):                                                                                                                      
         inputs = self.sam2_processor(                                                                                                                                       
             images=image,                                                                                                                                                   
             input_boxes=[boxes_xyxy],  # All boxes at once                                                                                                                  
             return_tensors="pt"                                                                                                                                             
         ).to(self.device)                                                                                                                                                   
                                                                                                                                                                             
         with torch.no_grad():                                                                                                                                               
             outputs = self.sam2_model(**inputs, multimask_output=False)                                                                                                     
                                                                                                                                                                             
     # Process all masks at once                                                                                                                                             
     masks = outputs.pred_masks.squeeze(1)                                                                                                                                   
     return [self._process_mask(m) for m in masks]                                                                                                                           
                                                                                                                                                                             
 ---                                                                                                                                                                         
 Phase 3: Tracker Optimizations                                                                                                                                              
                                                                                                                                                                             
 3.1 Pre-compute Sums in Conditioned Assignment                                                                                                                              
                                                                                                                                                                             
 File: bananatracker/tracker/banana_tracker.py (lines 301-312)                                                                                                               
                                                                                                                                                                             
 Add before the nested loop:                                                                                                                                                 
 def conditioned_assignment(self, dists, max_cost, ...):                                                                                                                     
     dists_cp = np.copy(dists)                                                                                                                                               
                                                                                                                                                                             
     if prediction_mask is None or tracklet_mask_dict is None:                                                                                                               
         matches, u_track, u_detection = matching.linear_assignment(dists_cp, thresh=max_cost)                                                                               
         return matches, u_track, u_detection, dists_cp                                                                                                                      
                                                                                                                                                                             
     # PRE-COMPUTE: row and column counts (instead of recalculating in loop)                                                                                                 
     valid_mask = dists <= max_cost                                                                                                                                          
     row_counts = valid_mask.sum(axis=1)                                                                                                                                     
     col_counts = valid_mask.sum(axis=0)                                                                                                                                     
                                                                                                                                                                             
     # PRE-COMPUTE: unique mask IDs once                                                                                                                                     
     unique_mask_ids = set(np.unique(prediction_mask).tolist()[1:]) if prediction_mask is not None else set()                                                                
                                                                                                                                                                             
     # Now iterate with cached values                                                                                                                                        
     for i in range(dists_cp.shape[0]):                                                                                                                                      
         for j in range(dists_cp.shape[1]):                                                                                                                                  
             if valid_mask[i, j]:                                                                                                                                            
                 if not (row_counts[i] > 1 or col_counts[j] > 1):                                                                                                            
                     # ... rest unchanged                                                                                                                                    
                                                                                                                                                                             
 3.2 Vectorize Kalman Motion Covariance                                                                                                                                      
                                                                                                                                                                             
 File: bananatracker/tracker/kalman_filter.py (lines 189-194)                                                                                                                
                                                                                                                                                                             
 Replace loop with vectorized version:                                                                                                                                       
 # OLD (loop):                                                                                                                                                               
 motion_cov = []                                                                                                                                                             
 for i in range(len(mean)):                                                                                                                                                  
     motion_cov.append(np.diag(sqr[i]))                                                                                                                                      
 motion_cov = np.asarray(motion_cov)                                                                                                                                         
                                                                                                                                                                             
 # NEW (vectorized):                                                                                                                                                         
 N = len(mean)                                                                                                                                                               
 motion_cov = np.zeros((N, 8, 8), dtype=np.float64)                                                                                                                          
 np.einsum('ij,jk->ijk', sqr, np.eye(8), out=motion_cov)                                                                                                                     
 # Or simpler:                                                                                                                                                               
 idx = np.arange(8)                                                                                                                                                          
 motion_cov[:, idx, idx] = sqr                                                                                                                                               
                                                                                                                                                                             
 ---                                                                                                                                                                         
 Config Changes Summary                                                                                                                                                      
                                                                                                                                                                             
 File: bananatracker/config.py                                                                                                                                               
                                                                                                                                                                             
 # Detection                                                                                                                                                                 
 detection_conf_thresh: float = 0.4                                                                                                                                          
 detection_batch_size: int = 16      # NEW: batch size for YOLO                                                                                                              
 use_half_precision: bool = True      # NEW: FP16 inference                                                                                                                  
                                                                                                                                                                             
 # Tracker                                                                                                                                                                   
 track_thresh: float = 0.5                                                                                                                                                   
 track_buffer: int = 90                                                                                                                                                      
                                                                                                                                                                             
 # Camera Motion - Optimized ECC (user preference: best quality)                                                                                                             
 cmc_method: str = "ecc"              # Keep ECC for accuracy                                                                                                                
 ecc_max_iterations: int = 500        # Reduce from 5000 (still converges)                                                                                                   
 ecc_termination_eps: float = 1e-4    # Slightly relaxed for speed                                                                                                           
 ecc_downscale: int = 4               # Process at 1/4 resolution (faster)                                                                                                   
                                                                                                                                                                             
 # Masks                                                                                                                                                                     
 sam2_use_fp16: bool = True           # NEW: FP16 for SAM2                                                                                                                   
                                                                                                                                                                             
 ---                                                                                                                                                                         
 Files to Modify                                                                                                                                                             
 ┌──────────────────────────────────┬────────────────────────────────────────────┐                                                                                           
 │               File               │                  Changes                   │                                                                                           
 ├──────────────────────────────────┼────────────────────────────────────────────┤                                                                                           
 │ config.py                        │ Add batch_size, half_precision, ecc params │                                                                                           
 ├──────────────────────────────────┼────────────────────────────────────────────┤                                                                                           
 │ detector.py                      │ Add detect_batch(), enable FP16            │                                                                                           
 ├──────────────────────────────────┼────────────────────────────────────────────┤                                                                                           
 │ pipeline.py                      │ Add process_video_batched() method         │                                                                                           
 ├──────────────────────────────────┼────────────────────────────────────────────┤                                                                                           
 │ tracker/gmc.py                   │ Reduce ECC iterations to 500               │                                                                                           
 ├──────────────────────────────────┼────────────────────────────────────────────┤                                                                                           
 │ tracker/banana_tracker.py        │ Pre-compute sums in conditioned_assignment │                                                                                           
 ├──────────────────────────────────┼────────────────────────────────────────────┤                                                                                           
 │ tracker/kalman_filter.py         │ Vectorize motion_cov creation              │                                                                                           
 ├──────────────────────────────────┼────────────────────────────────────────────┤                                                                                           
 │ mask_propagation/mask_manager.py │ Batch SAM2, enable FP16                    │                                                                                           
 └──────────────────────────────────┴────────────────────────────────────────────┘                                                                                           
 ---                                                                                                                                                                         
 Verification                                                                                                                                                                
                                                                                                                                                                             
 1. Speed Test:                                                                                                                                                              
 import time                                                                                                                                                                 
 start = time.time()                                                                                                                                                         
 pipeline.process_video("test.mp4")                                                                                                                                          
 print(f"FPS: {total_frames / (time.time() - start)}")                                                                                                                       
                                                                                                                                                                             
 2. GPU Utilization:                                                                                                                                                         
 import torch                                                                                                                                                                
 print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.1f}GB")                                                                                                           
                                                                                                                                                                             
 3. Quality Check:                                                                                                                                                           
 - Compare track outputs before/after                                                                                                                                        
 - Verify no increase in ID switches                                                                                                                                         
 - Check detection counts are similar                                                                                                                                        
                                                                                                                                                                             
 ---                                                                                                                                                                         
 Expected Results                                                                                                                                                            
 ┌─────────────────────┬────────────┬───────────┐                                                                                                                            
 │       Metric        │   Before   │   After   │                                                                                                                            
 ├─────────────────────┼────────────┼───────────┤                                                                                                                            
 │ GPU Memory          │ 7GB        │ 20-30GB   │                                                                                                                            
 ├─────────────────────┼────────────┼───────────┤                                                                                                                            
 │ FPS                 │ ~5-10      │ ~30-50    │                                                                                                                            
 ├─────────────────────┼────────────┼───────────┤                                                                                                                            
 │ Detection Batching  │ 1 frame    │ 16 frames │                                                                                                                            
 ├─────────────────────┼────────────┼───────────┤                                                                                                                            
 │ Precision           │ FP32       │ FP16      │                                                                                                                            
 ├─────────────────────┼────────────┼───────────┤                                                                                                                            
 │ ECC Iterations      │ 5000       │ 500       │                                                                                                                            
 ├─────────────────────┼────────────┼───────────┤                                                                                                                            
 │ ECC Downscale       │ 2x         │ 4x        │                                                                                                                            
 ├─────────────────────┼────────────┼───────────┤                                                                                                                            
 │ SAM2 Box Processing │ Sequential │ Batched   │                                                                                                                            
 └─────────────────────┴────────────┴───────────┘                                                                                                                            
 Implementation Order (for Modal notebook)                                                                                                                                   
                                                                                                                                                                             
 1. detector.py - Add detect_batch() + FP16 (BIGGEST impact)                                                                                                                 
 2. gmc.py - Reduce ECC iterations to 500, increase downscale to 4                                                                                                           
 3. pipeline.py - Add process_video_batched() with frame buffer                                                                                                              
 4. mask_manager.py - Batch SAM2 boxes + FP16                                                                                                                                
 5. banana_tracker.py - Pre-compute sums (minor speedup)                                                                                                                     
 6. kalman_filter.py - Vectorize motion_cov (minor speedup)   