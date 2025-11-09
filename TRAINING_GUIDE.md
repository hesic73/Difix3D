# Training Difix3D From Scratch

This guide shows you how to train Difix3D from scratch with sparse views (e.g., 4 frames from a 30fps video).

## Quick Start

```bash
# 1. Prepare your data in COLMAP format
# 2. Run training with progressive Difix refinement
python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir data/my_scene \
    --result_dir outputs/my_scene \
    --data_factor 4 \
    --max_steps 30000
```

That's it! The pretrained Difix model will automatically enhance your 3D reconstruction during training.

---

## Data Preparation

### Expected Structure

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
    ├── 000000.png  # Frame 0 (t=0/30 sec)
    ├── 000009.png  # Frame 9 (t=9/30 sec)
    ├── 000019.png  # Frame 19 (t=19/30 sec)
    └── 000029.png  # Frame 29 (t=29/30 sec)
```

### Option 1: Generate COLMAP from Images

If you only have images without camera poses:

```bash
# Install COLMAP (Ubuntu)
sudo apt install colmap

# Or Mac
brew install colmap

# Run automatic reconstruction
colmap automatic_reconstructor \
    --workspace_path data/my_scene/colmap \
    --image_path data/my_scene/images
```

### Option 2: Use Nerfstudio Processing

```bash
# Install nerfstudio
pip install nerfstudio

# Process data
ns-process-data images \
    --data data/my_scene/raw_images \
    --output-dir data/my_scene

# This creates COLMAP data automatically
```

---

## Training Configuration

### Basic Training (Default Settings)

```bash
python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir data/my_scene \
    --result_dir outputs/my_scene \
    --max_steps 30000
```

### Customized Training

```bash
python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir data/my_scene \
    --result_dir outputs/my_scene \
    --data_factor 4 \
    --max_steps 30000 \
    --save_steps 10000 20000 30000 \
    --eval_steps 10000 20000 30000 \
    --fix_steps 3000 6000 9000 12000 15000 18000 21000 24000 27000 30000 \
    --test_every 0
```

### Important Parameters

| Parameter | Description | Recommended Value |
|-----------|-------------|-------------------|
| `--data_factor` | Image downscale factor | 4 (adjust for GPU) |
| `--max_steps` | Total training steps | 30000 (30-40 min) |
| `--fix_steps` | When to apply Difix refinement | Every 3000 steps |
| `--test_every` | Test set sampling (0=use all for training) | 0 for 4 frames |
| `--save_steps` | When to save checkpoints | 10000 20000 30000 |
| `--eval_steps` | When to evaluate | 10000 20000 30000 |

### GPU Memory Optimization

If you run out of memory:

```bash
# Option 1: Increase downscale factor
--data_factor 8  # or even 16

# Option 2: Reduce batch size
--batch_size 1

# Option 3: Use packed mode
--packed
```

---

## What Happens During Training

### Timeline

```
Step 0-3000:
  ├─ Train 3DGS on your 4 frames
  └─ Initial reconstruction

Step 3000 (First Difix refinement):
  ├─ Render novel views from current model
  ├─ Apply pretrained Difix to fix artifacts
  ├─ Add fixed images to training set
  └─ Now training with 4 original + N fixed views

Step 3000-6000:
  ├─ Continue training with expanded data
  └─ 3DGS improves with more views

Step 6000 (Second Difix refinement):
  ├─ Render new novel views
  ├─ Fix with Difix (better than step 3000!)
  └─ Add to training

... (repeats at each fix_step)

Step 30000:
  └─ Final model with progressive improvements
```

### Output Files

During training, you'll get:

```
outputs/my_scene/
├── ckpts/
│   ├── ckpt_10000_rank0.pt
│   ├── ckpt_20000_rank0.pt
│   └── ckpt_29999_rank0.pt  # Final checkpoint
├── renders/
│   ├── novel/3000/  # Novel views at each fix step
│   │   ├── Pred/    # Raw renders
│   │   ├── Fixed/   # Difix-enhanced
│   │   └── Ref/     # Reference images
│   └── val/10000/   # Validation renders at eval steps
│       ├── GT/
│       └── Pred/
├── stats/
│   └── val_step10000.json  # Metrics (PSNR, SSIM, LPIPS)
└── tb/  # TensorBoard logs
```

---

## Monitoring Training

### Check Progress

```bash
# View metrics
cat outputs/my_scene/stats/val_step*.json

# Or use TensorBoard
tensorboard --logdir outputs/my_scene/tb
```

### Expected Metrics

| Metric | Step 10000 | Step 20000 | Step 30000 |
|--------|------------|------------|------------|
| PSNR | ~20-25 dB | ~23-28 dB | ~25-30 dB |
| SSIM | ~0.7-0.8 | ~0.8-0.9 | ~0.85-0.95 |
| LPIPS | ~0.3-0.4 | ~0.2-0.3 | ~0.15-0.25 |

*(Actual values depend on scene complexity and view sparsity)*

---

## Rendering Video

### Option 1: Use Built-in Rendering

The training script automatically renders novel views at each fix step:

```bash
# Check rendered videos at:
outputs/my_scene/renders/novel/30000/Fixed/
```

### Option 2: Custom Rendering

```python
import torch
from pathlib import Path
from examples.gsplat.simple_trainer_difix3d import Runner, Config
from gsplat.strategy import DefaultStrategy
from datasets.traj import generate_interpolated_path

# Load trained model
cfg = Config(
    data_dir="data/my_scene",
    data_factor=4,
    result_dir="outputs/my_scene",
    normalize_world_space=False,
    test_every=0,
    strategy=DefaultStrategy(verbose=False),
    disable_viewer=True,
)
runner = Runner(0, 0, 1, cfg)

# Load checkpoint
ckpt = torch.load("outputs/my_scene/ckpts/ckpt_29999_rank0.pt")
for k in runner.splats.keys():
    runner.splats[k].data = ckpt['splats'][k]

# Interpolate camera poses (4 frames → 30 frames)
train_poses = runner.parser.camtoworlds[runner.trainset.indices]
interpolated_poses = generate_interpolated_path(train_poses, n_interp=8)

# Render
runner.render_traj(step=30000, camtoworlds_all=interpolated_poses, tag="video")
```

### Option 3: Add Final Enhancement (Difix3D+)

For best quality, apply Difix post-processing:

```bash
python src/inference_difix.py \
    --model_name "nvidia/difix_ref" \
    --input_image "outputs/my_scene/renders/video/30000/Pred" \
    --ref_image "data/my_scene/images" \
    --prompt "remove degradation" \
    --output_dir "outputs/my_scene/final" \
    --timestep 199

# Create video
ffmpeg -r 30 -i "outputs/my_scene/final/%04d.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 \
    "outputs/my_scene/video.mp4"
```

---

## Alternative: Nerfstudio Backend

If you prefer using Nerfstudio:

### Install Plugin

```bash
cd examples/nerfstudio
pip install -e .
cd ../..
```

### Train

```bash
ns-train difix3d \
    --data data/my_scene \
    --output-dir outputs/my_scene_ns \
    --max-num-iterations 30000 \
    --pipeline.model.appearance-embed-dim 0 \
    --pipeline.model.camera-optimizer.mode off \
    nerfstudio-data \
        --orientation-method none \
        --center-method none \
        --auto-scale-poses False \
        --downscale-factor 4 \
        --eval-mode filename
```

### Render

```bash
# Create camera path
ns-render camera-path \
    --load-config outputs/my_scene_ns/difix3d/*/config.yml \
    --camera-path-filename camera_path.json \
    --output-path outputs/my_scene_ns/video.mp4
```

---

## Tips for Best Results

### 1. Camera Pose Quality

Good COLMAP reconstruction is crucial:

```bash
# Check COLMAP quality
colmap gui --database_path data/my_scene/colmap/database.db \
           --image_path data/my_scene/images
```

If COLMAP fails:
- Use more frames (6-8 instead of 4)
- Ensure sufficient overlap between views
- Check image quality (not too blurry)
- Try different feature extractors (SIFT, SuperPoint, etc.)

### 2. Training Duration

```bash
# Quick test (10k steps, ~10 min)
--max_steps 10000 --fix_steps 3000 6000 9000

# Standard (30k steps, ~30 min)
--max_steps 30000 --fix_steps 3000 6000 ... 30000

# High quality (60k steps, ~1 hour)
--max_steps 60000 --fix_steps 3000 6000 ... 60000
```

### 3. Sparse View Handling

For very sparse views (3-4 frames):

```bash
# Use all images for training
--test_every 0

# More frequent Difix refinement
--fix_steps 2000 4000 6000 8000 ...

# Train longer
--max_steps 60000
```

### 4. Scene-Specific Tuning

For outdoor scenes:
```bash
--no-normalize-world-space
```

For indoor/bounded scenes:
```bash
--normalize-world-space
```

For large scenes:
```bash
--global-scale 2.0  # or adjust as needed
```

---

## Troubleshooting

### Problem: Training is slow

**Solution:**
```bash
# Reduce resolution
--data_factor 8

# Reduce fix frequency
--fix_steps 5000 10000 15000 20000 25000 30000

# Skip some eval steps
--eval_steps 30000
```

### Problem: Poor novel view quality

**Solution:**
- Check COLMAP pose accuracy
- Train longer (60k+ steps)
- More frequent fixes
- Apply Difix3D+ post-processing
- Use more training frames (6-8 instead of 4)

### Problem: Artifacts in renders

**Solution:**
```bash
# More aggressive artifact removal
--fix_steps 2000 4000 6000 ...

# Apply final enhancement
python enhance_existing_video.py \
    --input outputs/my_scene/renders/video/30000/Pred \
    --output outputs/my_scene/enhanced \
    --use-ref --ref-images data/my_scene/images
```

### Problem: Out of memory

**Solution:**
```bash
# Downscale more aggressively
--data_factor 16

# Reduce batch size
--batch_size 1

# Use packed mode
--packed
```

---

## Complete Example

```bash
# Step 1: Prepare data
mkdir -p data/my_scene/images
cp frame_0.png data/my_scene/images/000000.png
cp frame_9.png data/my_scene/images/000009.png
cp frame_19.png data/my_scene/images/000019.png
cp frame_29.png data/my_scene/images/000029.png

# Step 2: Generate COLMAP
colmap automatic_reconstructor \
    --workspace_path data/my_scene/colmap \
    --image_path data/my_scene/images

# Step 3: Train Difix3D
python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir data/my_scene \
    --result_dir outputs/my_scene \
    --data_factor 4 \
    --max_steps 30000 \
    --test_every 0

# Step 4: Check results
ls outputs/my_scene/renders/novel/30000/Fixed/
# Final enhanced frames are here!

# Step 5: Create video
ffmpeg -r 30 -i "outputs/my_scene/renders/novel/30000/Fixed/%04d.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 \
    "outputs/my_scene/video.mp4"
```

---

## Summary

**Key Points:**

1. **Difix model is pretrained** - Just download and use, no training needed
2. **3DGS needs per-scene training** - ~30 minutes on RTX 3090
3. **Progressive refinement** - Difix automatically improves model during training
4. **fix_steps are crucial** - When Difix enhances novel views and adds them back
5. **Final enhancement optional** - Apply Difix3D+ for even better quality

**Expected Quality:**

- Without Difix3D: Sparse view 3DGS (artifacts, floaters)
- With Difix3D training: Much better (progressive improvement)
- With Difix3D+ post: Best quality (final polish)

**Time Investment:**

- Data preparation: 5-10 min
- Training: 30-40 min (30k steps)
- Rendering + enhancement: 5-10 min
- **Total: ~45-60 minutes**

Good luck with your training!
