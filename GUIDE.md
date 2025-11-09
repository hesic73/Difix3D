# Difix3D Complete Guide

## Understanding the Paper

This paper proposes using **single-step diffusion models** to improve 3D reconstruction quality. It has three key components:

### 1. **Difix** - Pretrained Single-Step Diffusion Model

**Problem:** 3DGS/NeRF renderings often have artifacts (blur, floaters, missing details)

**Solution:** A single-step diffusion model that takes degraded images and outputs clean ones

**Key Points:**
- ✅ **Pretrained** - Download from HuggingFace, ready to use
- ⚡ **Single-step inference** - Much faster than traditional diffusion (50-1000 steps)
- 🎯 Trained specifically on 3D reconstruction artifacts
- 📷 Supports reference image guidance (`nvidia/difix_ref`)

```
Degraded rendering → [Difix] → Clean image
```

### 2. **Difix3D** - Progressive 3D Training Strategy

**Problem:** Sparse views (e.g., 4 frames) produce poor novel view quality in 3DGS

**Solution:** During training, periodically:
1. Render novel views
2. Fix them with Difix
3. Add fixed images back to training set
4. Continue training with expanded data

**Effect:** Self-improvement loop - the 3D model gets progressively better

```
Training loop:
4 frames → Train 3DGS (3k steps) → Render novel views → Difix fix → Add to training
                ↑                                                        ↓
                └──────────────── Continue training ←─────────────────────┘
                (Repeat multiple times, 3DGS improves each iteration)
```

### 3. **Difix3D+** - Real-time Post-Processing

**Problem:** Even with Difix3D training, rendered images may still have blur or missing details

**Solution:** Apply Difix as final post-processing on rendered frames

**Key Points:**
- ✅ **No retraining needed**
- ✅ Works on any existing 3DGS/NeRF model
- ⚡ Real-time (single-step diffusion is fast)

```
Your case:
Trained model → Render frames → [Difix post-process] → Enhanced video
```

---

## Two Usage Scenarios

### Scenario A: You Have an Existing Model (Post-processing)

**Your situation:** Already trained splatfacto/3DGS model and rendered video

**Solution:** Use **Difix3D+** (post-processing only)
- ❌ No retraining needed
- ⚡ 5-10 minutes to enhance video
- ✨ Removes artifacts, improves details

**See:** [Quick Start - Post-Processing](#post-processing-existing-models)

### Scenario B: Training From Scratch (Full Pipeline)

**Your situation:** Have 4 sparse frames, want to train from scratch

**Solution:** Use full **Difix3D** training pipeline
- ✅ Train 3DGS with progressive refinement
- 🔄 Difix automatically improves model during training
- ✨ Better geometry and appearance quality

**See:** [Training From Scratch](#training-from-scratch)

---

## Post-Processing Existing Models

### Quick Start

```bash
# Basic enhancement
python enhance_existing_video.py \
    --input your_video.mp4 \
    --output enhanced.mp4

# With reference images (better quality)
python enhance_existing_video.py \
    --input your_video.mp4 \
    --output enhanced.mp4 \
    --use-ref \
    --ref-images /path/to/training/images
```

### Using Python API

```python
from pipeline_difix import DifixPipeline
from PIL import Image
from glob import glob
from tqdm import tqdm
import os

# Load pretrained Difix model
pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
pipe.to("cuda")

# Process each frame
input_frames = sorted(glob("rendered_frames/*.png"))
output_dir = "enhanced_frames"
os.makedirs(output_dir, exist_ok=True)

for i, frame_path in enumerate(tqdm(input_frames)):
    image = Image.open(frame_path).convert('RGB')

    # Single-step enhancement
    enhanced = pipe(
        prompt="remove degradation",
        image=image,
        num_inference_steps=1,
        timesteps=[199],
        guidance_scale=0.0
    ).images[0]

    enhanced.save(f"{output_dir}/{i:04d}.png")
```

### With Reference Images (Recommended)

```python
pipe = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
pipe.to("cuda")

ref_images = sorted(glob("training_images/*.png"))

for i, frame_path in enumerate(tqdm(input_frames)):
    image = Image.open(frame_path).convert('RGB')

    # Select closest reference image
    ref_idx = min(i % len(ref_images), len(ref_images) - 1)
    ref_image = Image.open(ref_images[ref_idx]).convert('RGB')

    enhanced = pipe(
        prompt="remove degradation",
        image=image,
        ref_image=ref_image,
        num_inference_steps=1,
        timesteps=[199],
        guidance_scale=0.0
    ).images[0]

    enhanced.save(f"{output_dir}/{i:04d}.png")
```

---

## Training From Scratch

### Data Preparation

Your data should be in COLMAP format:

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
    ├── 000000.png  # Frame 0
    ├── 000009.png  # Frame 9
    ├── 000019.png  # Frame 19
    └── 000029.png  # Frame 29
```

**Generate COLMAP data:**

```bash
# Option 1: COLMAP automatic reconstruction
colmap automatic_reconstructor \
    --workspace_path data/my_scene/colmap \
    --image_path data/my_scene/images

# Option 2: Nerfstudio data processing
ns-process-data images \
    --data data/my_scene/raw_images \
    --output-dir data/my_scene
```

### Method 1: Using gsplat (Recommended)

```bash
SCENE_ID="my_scene"
DATA_DIR="data/${SCENE_ID}"
OUTPUT_DIR="outputs/difix3d/gsplat/${SCENE_ID}"

python examples/gsplat/simple_trainer_difix3d.py default \
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

- **Steps 0-3000:** Initial 3DGS training on your 4 frames
- **Step 3000:** First Difix refinement - renders novel views, fixes artifacts, adds to training
- **Steps 3000-6000:** Continue training with expanded dataset
- **Step 6000:** Second Difix refinement
- ... (repeats at each fix_step)
- **Step 30000:** Final model with progressive improvements

**Key parameters:**

- `--data_factor 4`: Downsample images by 4x (adjust for GPU memory)
- `--test_every 1`: Use every Nth image for testing (0 = use all for training)
- `--fix_steps`: When to apply Difix refinement
- `--max_steps`: Total training steps

### Method 2: Using Nerfstudio

```bash
# Install Difix3D nerfstudio plugin
cd examples/nerfstudio
pip install -e .
cd ../..

# Train with Difix3D
SCENE_ID="my_scene"
DATA="data/${SCENE_ID}"
OUTPUT_DIR="outputs/difix3d/nerfacto/${SCENE_ID}"

ns-train difix3d \
    --data ${DATA} \
    --pipeline.model.appearance-embed-dim 0 \
    --pipeline.model.camera-optimizer.mode off \
    --output-dir ${OUTPUT_DIR} \
    --max-num-iterations 30000 \
    nerfstudio-data \
        --orientation-method none \
        --center-method none \
        --auto-scale-poses False \
        --downscale-factor 4 \
        --eval-mode filename
```

**Optional: Load from existing checkpoint**

```bash
# If you have a pretrained nerfacto model
CKPT_PATH="path/to/nerfstudio_models/step-000029999.ckpt"

ns-train difix3d \
    --data ${DATA} \
    --load-checkpoint ${CKPT_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --max-num-iterations 30000 \
    nerfstudio-data --downscale-factor 4
```

### Rendering Video

After training, render your video:

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

# Initialize runner
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

# Generate interpolated camera poses
# 4 frames → 30 frames (interpolate ~8 frames between each pair)
train_poses = runner.parser.camtoworlds[runner.trainset.indices]
interpolated_poses = generate_interpolated_path(train_poses, n_interp=8)

# Render
runner.render_traj(step=30000, camtoworlds_all=interpolated_poses, tag="video")
# Output: outputs/difix3d/gsplat/my_scene/renders/video/30000/Pred/
```

### Optional: Apply Difix3D+ Post-Processing

For even better quality, apply final enhancement:

```bash
python src/inference_difix.py \
    --model_name "nvidia/difix_ref" \
    --input_image "outputs/difix3d/gsplat/my_scene/renders/video/30000/Pred" \
    --ref_image "data/my_scene/images" \
    --prompt "remove degradation" \
    --output_dir "outputs/difix3d/gsplat/my_scene/final_enhanced" \
    --timestep 199

# Create final video
ffmpeg -r 30 -i "outputs/difix3d/gsplat/my_scene/final_enhanced/%04d.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 \
    "outputs/difix3d/gsplat/my_scene/output_30fps.mp4"
```

---

## Method Comparison

| Aspect | Post-Processing (Difix3D+) | Train From Scratch (Difix3D) |
|--------|---------------------------|------------------------------|
| **Retraining needed?** | ❌ No | ✅ Yes (~30 min) |
| **Works on existing models?** | ✅ Yes | ❌ No |
| **Processing time** | ⚡ 5-10 min | 🐢 30 min - 1 hour |
| **Improves 3D geometry?** | ❌ No (image-only) | ✅ Yes |
| **Quality** | ✨ Good | ✨✨ Better |
| **Use case** | Quick enhancement | Best quality from scratch |

---

## Performance Benchmarks

On RTX 3090:

| Task | Time |
|------|------|
| Load Difix model | ~5 seconds |
| Process 1 frame (no ref) | ~0.5 seconds |
| Process 1 frame (with ref) | ~0.8 seconds |
| Process 100 frames | ~1-2 minutes |
| Process 1000 frames | ~10-15 minutes |
| Train Difix3D (30k steps) | ~30-40 minutes |

---

## Troubleshooting

### COLMAP fails or produces bad camera poses

**Solution:**
- Use more frames (8-10 instead of 4)
- Check image quality and overlap
- Try Nerfstudio's data processing
- Use SfM alternatives (HLOC, etc.)

### Out of GPU memory

**Solution:**
```bash
# Reduce image resolution
--data_factor 8

# Reduce batch size
--batch_size 1

# For post-processing, resize images
image = image.resize((image.width // 2, image.height // 2))
```

### Training produces artifacts

**Solution:**
- Train longer: `--max_steps 60000`
- More fix steps: `--fix_steps 2000 4000 6000 ...`
- Apply Difix3D+ post-processing
- Check COLMAP quality

### Video not smooth

**Solution:**
- Increase interpolation: `n_interp=10` or higher
- Use smooth camera path
- Apply motion blur post-processing
- Check camera pose quality

---

## Technical Details

### How Difix Works

```
Architecture: Based on SD-Turbo (single-step distilled Stable Diffusion)
Training Data: Degraded 3DGS/NeRF renders → Ground truth pairs
Input: Degraded rendering + (optional) reference image
Output: Clean image
Inference Time: ~0.5-1 sec/frame (single step!)
```

### Why Single-Step is Enough

Traditional diffusion starts from pure noise (needs many steps).
Difix starts from "almost correct but degraded" (needs only one correction step).
Achieved through distillation from multi-step diffusion.

### Reference Image Usage

```
Without reference:
  Model only sees rendering → guesses artifacts
  ↓
  May hallucinate inconsistent details

With reference:
  Model sees:
  1. Rendering (knows geometry and viewpoint)
  2. Reference (knows real textures and details)
  ↓
  More accurate recovery, maintains 3D consistency
```

---

## Summary

### For existing models (you have trained splatfacto):
✅ Use **Difix3D+** post-processing - no retraining needed

### For training from scratch (you have sparse frames):
✅ Use **Difix3D** full pipeline - progressive improvement during training

### Key Points:
- Difix model is **pretrained** - just download and use
- 3D reconstruction (3DGS) needs **per-scene training**
- Difix3D combines both with progressive refinement
- Expected time: ~30-40 min training + ~5 min enhancement
