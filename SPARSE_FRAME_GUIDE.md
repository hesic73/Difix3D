# Difix3D Guide: Sparse Frame Video Generation

## Your Use Case

**Input:** 4 frames extracted from a 1-second, 30fps video (frames 0, 9, 19, 29)
**Output:** Full 1-second 30fps video (30 frames)
**Data format:** COLMAP or Nerfstudio

---

## Answer to Your Questions

### Q1: Is this a pretrained diffusion model (inference only) or does it require per-scene training like 3DGS?

**Answer: BOTH**

The Difix3D system consists of two components:

1. **Difix (Diffusion Model)** - ✅ **PRETRAINED, inference only**
   - A single-step diffusion model for artifact removal
   - Downloaded from HuggingFace: `nvidia/difix` or `nvidia/difix_ref`
   - No training required
   - Used to enhance rendered images

2. **3D Reconstruction (3DGS/NeRF)** - ⚠️ **Requires per-scene training**
   - Like traditional 3D Gaussian Splatting
   - Must be trained on your specific scene (your 4 frames)
   - Training takes ~30,000 steps (varies by scene complexity)

3. **Difix3D (The Full System)** - 🔄 **Progressive training + inference**
   - Trains 3DGS on your sparse views
   - At certain steps, uses the pretrained Difix model to:
     - Render novel views
     - Fix artifacts with diffusion
     - Add fixed views back to training
   - This progressive refinement improves final quality

### Q2: How do I do inference for one scene?

See the workflow below! 👇

---

## Complete Workflow

### Prerequisites

```bash
# 1. Install dependencies
cd /home/user/Difix3D
pip install -r requirements.txt

# 2. Install gsplat
pip install gsplat

# 3. Prepare your data in COLMAP format
```

### Data Structure

Your data should look like this:

```
data/my_scene/
├── colmap/
│   └── sparse/
│       └── 0/
│           ├── cameras.bin
│           ├── images.bin
│           ├── points3D.bin
│           └── database.db
└── images/
    ├── image_train_000000.png  # Frame 0 (0/30 sec)
    ├── image_train_000009.png  # Frame 9 (9/30 sec)
    ├── image_train_000019.png  # Frame 19 (19/30 sec)
    └── image_train_000029.png  # Frame 29 (29/30 sec)
```

**How to get COLMAP data:**

If you only have images, run COLMAP first:

```bash
# Install COLMAP
# Ubuntu: sudo apt install colmap
# Mac: brew install colmap

# Run COLMAP automatic reconstruction
colmap automatic_reconstructor \
    --workspace_path data/my_scene/colmap \
    --image_path data/my_scene/images
```

Or use Nerfstudio's data processing:

```bash
ns-process-data images --data data/my_scene/raw_images --output-dir data/my_scene
```

---

## Method 1: Simple Script (Recommended for Beginners)

```bash
# Run the complete workflow
python inference_example.py \
    --data_dir data/my_scene \
    --output_dir outputs/my_scene \
    --fps 30 \
    --duration 1.0 \
    --max_steps 30000 \
    --enhance  # Optional: apply Difix3D+ post-processing
```

This will:
1. ✅ Verify your data structure
2. 🏋️ Train 3DGS on your 4 frames (with Difix3D progressive refinement)
3. 🎬 Render interpolated frames for 30fps video
4. ✨ (Optional) Enhance with pretrained Difix model
5. 🎥 Create final MP4 video

**To skip training and use existing checkpoint:**

```bash
python inference_example.py \
    --data_dir data/my_scene \
    --output_dir outputs/my_scene \
    --skip_training \
    --ckpt_path outputs/my_scene/ckpts/ckpt_29999_rank0.pt \
    --enhance
```

---

## Method 2: Step-by-Step (For Advanced Users)

### Step 1: Train 3DGS with Difix3D

```bash
SCENE_ID="my_scene"
DATA_DIR="data/${SCENE_ID}"
OUTPUT_DIR="outputs/difix3d/gsplat/${SCENE_ID}"

CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA_DIR} \
    --data_factor 4 \
    --result_dir ${OUTPUT_DIR} \
    --no-normalize-world-space \
    --test_every 1 \
    --max_steps 30000 \
    --save_steps 10000 20000 30000 \
    --eval_steps 10000 20000 30000 \
    --fix_steps 3000 6000 9000 12000 15000 18000 21000 24000 27000 30000
```

**What happens during training:**

- Steps 0-3000: Initial 3DGS training on your 4 frames
- Step 3000: **Difix refinement** - renders novel views, fixes artifacts, adds to training
- Steps 3000-6000: Continue training with more data
- Step 6000: **Difix refinement** again
- ... (repeats at each fix_step)
- Step 30000: Final model

**Key parameters:**

- `--data_factor 4`: Downsample images by 4x (adjust based on GPU memory)
- `--test_every 1`: Use every 1st image for testing (with 4 images, you might want to use all for training, set to 0)
- `--fix_steps`: When to apply Difix refinement

### Step 2: Render Novel Views

```python
import torch
from pathlib import Path
from examples.gsplat.simple_trainer_difix3d import Runner, Config
from gsplat.strategy import DefaultStrategy
from datasets.traj import generate_interpolated_path

# Setup
data_dir = "data/my_scene"
output_dir = "outputs/difix3d/gsplat/my_scene"
ckpt_path = f"{output_dir}/ckpts/ckpt_29999_rank0.pt"

# Initialize
cfg = Config(
    data_dir=data_dir,
    data_factor=4,
    result_dir=output_dir,
    normalize_world_space=False,
    test_every=1,
    strategy=DefaultStrategy(verbose=False),
    disable_viewer=True,
)
runner = Runner(0, 0, 1, cfg)

# Load checkpoint
ckpt = torch.load(ckpt_path, map_location=runner.device)
for k in runner.splats.keys():
    runner.splats[k].data = ckpt['splats'][k]

# Generate camera poses for 30fps (4 frames → 30 frames)
train_poses = runner.parser.camtoworlds[runner.trainset.indices]
# Interpolate 8-9 frames between each pair to get ~30 total frames
interpolated_poses = generate_interpolated_path(train_poses, n_interp=8)

# Render
runner.render_traj(step=30000, camtoworlds_all=interpolated_poses, tag="video")
# Output: outputs/difix3d/gsplat/my_scene/renders/video/30000/Pred/
```

### Step 3: (Optional) Apply Difix3D+ Enhancement

```bash
python src/inference_difix.py \
    --model_name "nvidia/difix_ref" \
    --input_image "outputs/difix3d/gsplat/my_scene/renders/video/30000/Pred" \
    --ref_image "data/my_scene/images" \
    --prompt "remove degradation" \
    --output_dir "outputs/difix3d/gsplat/my_scene/enhanced" \
    --timestep 199
```

### Step 4: Create Video

```bash
ffmpeg -r 30 -i "outputs/difix3d/gsplat/my_scene/enhanced/%04d.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 \
    "outputs/difix3d/gsplat/my_scene/output_30fps.mp4"
```

---

## Method 3: Using Nerfstudio (Alternative)

If you prefer Nerfstudio:

```bash
SCENE_ID="my_scene"
DATA="data/${SCENE_ID}"
OUTPUT_DIR="outputs/difix3d/nerfacto/${SCENE_ID}"

# First train base nerfacto model
ns-train nerfacto \
    --data ${DATA} \
    --output-dir ${OUTPUT_DIR}_base \
    --max-num-iterations 30000

# Then run Difix3D refinement
CKPT_PATH="${OUTPUT_DIR}_base/nerfacto/*/nerfstudio_models/step-000029999.ckpt"

ns-train difix3d \
    --data ${DATA} \
    --pipeline.model.appearance-embed-dim 0 \
    --pipeline.model.camera-optimizer.mode off \
    --output_dir ${OUTPUT_DIR} \
    --load-checkpoint ${CKPT_PATH} \
    --max_num_iterations 30000 \
    nerfstudio-data --orientation-method none --center_method none
```

---

## Understanding the Components

### 1. **Difix** (Pretrained Diffusion Model)

- **What:** Single-step diffusion model for artifact removal
- **Training:** ❌ No training needed (pretrained)
- **Usage:** Direct inference via `DifixPipeline` or `inference_difix.py`
- **Model:** `nvidia/difix` or `nvidia/difix_ref` (with reference image support)

```python
from pipeline_difix import DifixPipeline

pipe = DifixPipeline.from_pretrained("nvidia/difix_ref")
output = pipe(
    prompt="remove degradation",
    image=input_image,
    ref_image=reference_image,
    num_inference_steps=1,
    timesteps=[199]
).images[0]
```

### 2. **3DGS/NeRF** (3D Reconstruction)

- **What:** 3D scene representation
- **Training:** ✅ Per-scene training required (~30k steps)
- **Input:** Your 4 sparse frames + COLMAP camera poses
- **Output:** 3D model that can render novel views

### 3. **Difix3D** (Progressive 3D Update)

- **What:** Training strategy that uses Difix to improve 3DGS
- **How:**
  1. Train 3DGS on sparse views
  2. Periodically render novel views
  3. Use Difix to fix artifacts
  4. Add fixed views back to training data
  5. Repeat
- **Result:** Better 3D reconstruction from sparse views

### 4. **Difix3D+** (Post-rendering Enhancement)

- **What:** Apply Difix as final post-processing
- **When:** After rendering all frames from trained 3DGS
- **Why:** Further enhance quality by removing remaining artifacts

---

## Tips for Best Results

### 1. Data Quality
- Ensure your 4 frames have good camera pose estimates from COLMAP
- If COLMAP fails, you might need more frames or better features
- Check that your frames cover the scene well (not all from same angle)

### 2. Training Parameters

```bash
# For faster testing (lower quality):
--max_steps 10000 --data_factor 8

# For better quality (slower):
--max_steps 60000 --data_factor 2

# GPU memory issues:
--data_factor 8  # or higher
--batch_size 1
```

### 3. Interpolation

For smooth video, ensure camera path interpolation is smooth:

```python
# Linear interpolation (simpler)
from datasets.traj import generate_interpolated_path
poses = generate_interpolated_path(train_poses, n_interp=8)

# For more complex paths
from datasets.traj import generate_spiral_path
poses = generate_spiral_path(train_poses, ...)
```

### 4. Difix Enhancement

- Use `nvidia/difix_ref` (with reference images) for better quality
- Reference images should be closest training views to rendered views
- Timestep 199 is recommended (single-step diffusion)

---

## Expected Results

From 4 sparse frames to 30fps video:

```
Input:   4 frames (0, 9, 19, 29)
         ↓
3DGS:    Train 3D model (~30k steps, ~30 mins on RTX 3090)
         ↓
Difix3D: Progressive refinement (automatic during training)
         ↓
Render:  30 interpolated frames
         ↓
Difix3D+: Post-enhancement (optional, ~1 min for 30 frames)
         ↓
Output:  30fps video, 1 second duration
```

**Quality:**
- Without Difix3D: Some artifacts due to sparse training views
- With Difix3D: Cleaner novel views, progressive improvement
- With Difix3D+: Further enhanced, sharper details

---

## Troubleshooting

### Issue: COLMAP fails or produces bad camera poses

**Solution:**
- Use more frames (e.g., 8-10 instead of 4)
- Check image quality and overlap
- Try structure-from-motion alternatives (Nerfstudio's data processing)

### Issue: Out of GPU memory

**Solution:**
```bash
--data_factor 8  # Downsample more
--batch_size 1
```

### Issue: Artifacts in rendered views

**Solution:**
- Train longer: `--max_steps 60000`
- More fix steps: `--fix_steps 2000 4000 6000 ...`
- Apply Difix3D+ post-processing: `--enhance`

### Issue: Video is not smooth

**Solution:**
- Increase interpolation: `n_interp=10` or higher
- Use smoother camera path interpolation
- Apply motion blur post-processing

---

## Quick Start Example

```bash
# 1. Prepare data
mkdir -p data/my_scene/images
cp frame_0000.png data/my_scene/images/image_train_000000.png
cp frame_0009.png data/my_scene/images/image_train_000009.png
cp frame_0019.png data/my_scene/images/image_train_000019.png
cp frame_0029.png data/my_scene/images/image_train_000029.png

# 2. Run COLMAP
colmap automatic_reconstructor \
    --workspace_path data/my_scene/colmap \
    --image_path data/my_scene/images

# 3. Run Difix3D
python inference_example.py \
    --data_dir data/my_scene \
    --output_dir outputs/my_scene \
    --fps 30 \
    --duration 1.0 \
    --enhance

# 4. Done!
# Output: outputs/my_scene/output_30fps.mp4
```

---

## Summary

| Component | Pretrained? | Training Needed? | Purpose |
|-----------|-------------|------------------|---------|
| **Difix** | ✅ Yes | ❌ No | Artifact removal (inference only) |
| **3DGS** | ❌ No | ✅ Yes (per-scene) | 3D reconstruction |
| **Difix3D** | Hybrid | ✅ Yes (uses pretrained Difix during training) | Progressive 3D refinement |
| **Difix3D+** | ✅ Yes | ❌ No | Post-rendering enhancement |

**For your use case (4 frames → 30fps video):**

1. **You MUST train** the 3DGS model on your scene (per-scene training)
2. **You DON'T train** the Difix diffusion model (use pretrained)
3. **Difix3D combines both** for best results
4. **Expected time:** ~30 mins training + ~1 min rendering on RTX 3090
