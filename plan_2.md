Plan: Integrate SAM2.1 + Cutie for Module 2 (Mask Enhancement)                                                                                                               
                                                                                                                                                                              
 Overview           

The architecture is:                                                                                                                                                        
  - SAM2.1 → Creates initial masks from bounding boxes (one-time per new tracklet)                                                                                            
  - Cutie → Propagates those masks temporally across all subsequent frames                                                                                                                                                                
                                                                                                                                                                              
 Replace SAM1 with SAM2.1 (via HuggingFace) for initial mask creation while keeping Cutie for temporal mask propagation in the McByte tracker.                                
                                                                                                                                                                              
 Current Architecture Analysis                                                                                                                                                
                                                                                                                                                                              
 Current Flow (SAM1 + Cutie)                                                                                                                                                  
                                                                                                                                                                              
 New Tracklet Created → SAM1 creates initial mask from bbox → Cutie propagates mask across frames                                                                             
                                                           ↓                                                                                                                  
                                     Mask metrics (mc/mf) used in cost matrix enrichment                                                                                      
                                                                                                                                                                              
 Key Files                                                                                                                                                                    
                                                                                                                                                                              
 - mask_propagation/mask_manager.py - Main file to modify (SAM initialization + inference)                                                                                    
 - yolox/tracker/mcbyte_tracker.py - Uses masks via conditioned_assignment() (NO changes needed)                                                                              
 - tools/demo_track.py - Entry point (minimal changes for CLI args)                                                                                                           
                                                                                                                                                                              
 Current SAM1 Usage (mask_manager.py)                                                                                                                                         
                                                                                                                                                                              
 # Initialization (lines 46-52)                                                                                                                                               
 from segment_anything import sam_model_registry, SamPredictor                                                                                                                
 sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)                                                                                                              
 sam_predictor = SamPredictor(sam)                                                                                                                                            
                                                                                                                                                                              
 # Per-image inference (lines 119, 147-152, 187, 235-240)                                                                                                                     
 self.sam_predictor.set_image(img)                                                                                                                                            
 masks, _, _ = self.sam_predictor.predict_torch(                                                                                                                              
     point_coords=None, point_labels=None,                                                                                                                                    
     boxes=transformed_boxes, multimask_output=False                                                                                                                          
 )                                                                                                                                                                            
                                                                                                                                                                              
 SAM2.1 Integration Plan                                                                                                                                                      
                                                                                                                                                                              
 Step 1: Update Imports and Initialization                                                                                                                                    
                                                                                                                                                                              
 Replace SAM1 imports with SAM2.1:                                                                                                                                            
 # OLD                                                                                                                                                                        
 from segment_anything import sam_model_registry, SamPredictor                                                                                                                
                                                                                                                                                                              
 # NEW                                                                                                                                                                        
 from transformers import Sam2Model, Sam2Processor                                                                                                                            
 import torch                                                                                                                                                                 
                                                                                                                                                                              
 Update __init__ to accept configuration:                                                                                                                                     
 def __init__(self, sam2_model_id="facebook/sam2.1-hiera-large",                                                                                                              
              cutie_weights_path=None, device="cuda:0", hf_token=None):                                                                                                       
     # SAM2.1 via HuggingFace                                                                                                                                                 
     self.sam2_model = Sam2Model.from_pretrained(sam2_model_id, token=hf_token)                                                                                               
     self.sam2_processor = Sam2Processor.from_pretrained(sam2_model_id, token=hf_token)                                                                                       
     self.sam2_model.to(device).eval()                                                                                                                                        
                                                                                                                                                                              
     # Cutie (unchanged)                                                                                                                                                      
     # ... existing Cutie initialization ...                                                                                                                                  
                                                                                                                                                                              
 Step 2: Create SAM2.1 Prediction Method                                                                                                                                      
                                                                                                                                                                              
 New method to replace SAM1 box-prompted segmentation:                                                                                                                        
 def _sam2_predict_boxes(self, image: np.ndarray, boxes_xyxy: list) -> np.ndarray:                                                                                            
     """                                                                                                                                                                      
     Generate masks for multiple bounding boxes using SAM2.1                                                                                                                  
                                                                                                                                                                              
     Args:                                                                                                                                                                    
         image: RGB image as numpy array (H, W, 3)                                                                                                                            
         boxes_xyxy: List of [x1, y1, x2, y2] bounding boxes                                                                                                                  
                                                                                                                                                                              
     Returns:                                                                                                                                                                 
         masks: Binary masks as numpy array (N, H, W)                                                                                                                         
     """                                                                                                                                                                      
     # Format boxes for SAM2.1: [[[x1, y1, x2, y2], [x1, y1, x2, y2], ...]]                                                                                                   
     input_boxes = [[box for box in boxes_xyxy]]                                                                                                                              
                                                                                                                                                                              
     # Process with SAM2.1                                                                                                                                                    
     inputs = self.sam2_processor(                                                                                                                                            
         images=image,                                                                                                                                                        
         input_boxes=input_boxes,                                                                                                                                             
         return_tensors="pt"                                                                                                                                                  
     ).to(self.device)                                                                                                                                                        
                                                                                                                                                                              
     with torch.no_grad():                                                                                                                                                    
         outputs = self.sam2_model(**inputs, multimask_output=False)                                                                                                          
                                                                                                                                                                              
     # Post-process masks                                                                                                                                                     
     masks = self.sam2_processor.post_process_masks(                                                                                                                          
         outputs.pred_masks.cpu(),                                                                                                                                            
         inputs["original_sizes"].cpu()                                                                                                                                       
     )[0]  # (N, 1, H, W) → squeeze to (N, H, W)                                                                                                                              
                                                                                                                                                                              
     return masks.squeeze(1).numpy().astype(np.uint8)                                                                                                                         
                                                                                                                                                                              
 Step 3: Update Mask Creation Methods                                                                                                                                         
                                                                                                                                                                              
 Update initialize_first_masks() (line 118-182):                                                                                                                              
 - Replace self.sam_predictor.set_image() call                                                                                                                                
 - Replace self.sam_predictor.predict_torch() with self._sam2_predict_boxes()                                                                                                 
 - Ensure mask format compatibility with Cutie                                                                                                                                
                                                                                                                                                                              
 Update add_new_masks() (line 185-278):                                                                                                                                       
 - Same changes as above                                                                                                                                                      
                                                                                                                                                                              
 Step 4: Update CLI Arguments (demo_track.py)                                                                                                                                 
                                                                                                                                                                              
 Add new argument (HF token via environment variable is preferred):                                                                                                           
 parser.add_argument("--sam2_model_id", default="facebook/sam2.1-hiera-large",                                                                                                
                     help="SAM2.1 HuggingFace model ID")                                                                                                                      
                                                                                                                                                                              
 Note: HF_TOKEN should be passed via environment variable or directly to MaskManager, not as CLI arg for security.                                                            
                                                                                                                                                                              
 Critical Integration Points                                                                                                                                                  
                                                                                                                                                                              
 1. Mask Format Compatibility with Cutie                                                                                                                                      
                                                                                                                                                                              
 - SAM2.1 outputs: (N, H, W) binary masks                                                                                                                                     
 - Cutie expects: Combined mask with unique values per object (H, W) where pixel value = object ID                                                                            
 - Conversion logic (already exists in current code, reuse):                                                                                                                  
 mask = np.zeros((H, W))                                                                                                                                                      
 for mi in range(len(masks)):                                                                                                                                                 
     current_mask = masks[mi].astype(int)                                                                                                                                     
     current_mask[current_mask > 0] = mi + 1                                                                                                                                  
     non_occupied = (mask == 0).astype(int)                                                                                                                                   
     mask += (current_mask * non_occupied)                                                                                                                                    
                                                                                                                                                                              
 2. Memory Management                                                                                                                                                         
                                                                                                                                                                              
 - SAM2.1 model should be kept in eval mode                                                                                                                                   
 - Use torch.no_grad() during inference                                                                                                                                       
 - Leverage existing torch.inference_mode() context in demo_track.py                                                                                                          
                                                                                                                                                                              
 3. No Changes to Cost Matrix Enrichment                                                                                                                                      
                                                                                                                                                                              
 - conditioned_assignment() in mcbyte_tracker.py remains unchanged                                                                                                            
 - mc/mf metrics calculation works on any mask format (H, W) with object IDs                                                                                                  
 - Thresholds remain: MIN_MASK_AVG_CONF=0.6, MIN_MM1=0.9, MIN_MM2=0.05                                                                                                        
                                                                                                                                                                              
 Implementation Checklist                                                                                                                                                     
                                                                                                                                                                              
 mask_propagation/mask_manager.py                                                                                                                                             
                                                                                                                                                                              
 - Update imports (remove segment_anything, add transformers)                                                                                                                 
 - Update __init__ signature to accept SAM2.1 config                                                                                                                          
 - Add SAM2.1 model initialization                                                                                                                                            
 - Add _sam2_predict_boxes() method                                                                                                                                           
 - Update initialize_first_masks() to use SAM2.1                                                                                                                              
 - Update add_new_masks() to use SAM2.1                                                                                                                                       
 - Remove SAM1 transform logic (SAM2.1 handles internally)                                                                                                                    
                                                                                                                                                                              
 tools/demo_track.py                                                                                                                                                          
                                                                                                                                                                              
 - Add --sam2_model_id argument                                                                                                                                               
 - Add --hf_token argument                                                                                                                                                    
 - Pass new args to MaskManager initialization                                                                                                                                
                                                                                                                                                                              
 Verification Plan                                                                                                                                                            
                                                                                                                                                                              
 1. Unit Test SAM2.1 Mask Generation                                                                                                                                          
                                                                                                                                                                              
 # Test mask creation from bounding boxes                                                                                                                                     
 mask_manager = MaskManager(sam2_model_id="facebook/sam2.1-hiera-large")                                                                                                      
 test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)                                                                                                        
 test_boxes = [[100, 100, 200, 200], [300, 300, 400, 400]]                                                                                                                    
 masks = mask_manager._sam2_predict_boxes(test_image, test_boxes)                                                                                                             
 assert masks.shape == (2, 480, 640)                                                                                                                                          
                                                                                                                                                                              
 2. Integration Test with Cutie                                                                                                                                               
                                                                                                                                                                              
 # Run on a short video/image sequence                                                                                                                                        
 python tools/demo_track.py \                                                                                                                                                 
     --demo image \                                                                                                                                                           
     --path <test_frames_path> \                                                                                                                                              
     --sam2_model_id facebook/sam2.1-hiera-large \                                                                                                                            
     --hf_token <your_token> \                                                                                                                                                
     --vis_type basic                                                                                                                                                         
                                                                                                                                                                                                                                                       
                                                                                                                                                                              
 Dependencies Changes                                                                                                                                                         
                                                                                                                                                                              
 Remove:                                                                                                                                                                      
 segment-anything  # Old SAM1 library                                                                                                                                         
                                                                                                                                                                              
 Add:                                                                                                                                                                         
 transformers>=4.35.0                                                                                                                                                         
 huggingface_hub                                                                                                                                                              
                                                                                                                                                                              
 Note: The user will provide HF_TOKEN for model access.                                                                                                                       
                                                                                                                                                                              
 Design Decisions (Confirmed)                                                                                                                                                 
                                                                                                                                                                              
 1. Fully replace SAM1 - Remove segment_anything dependency, cleaner code                                                                                                     
 2. Default model: facebook/sam2.1-hiera-large - Best quality for sports tracking                                                                                             
 3. HuggingFace-only loading - Simpler code, automatic caching with HF_TOKEN  