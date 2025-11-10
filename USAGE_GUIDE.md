# Difix3D Usage Guide

## What is Difix3D?

Difix3D uses a **pretrained single-step diffusion model** to improve 3D reconstruction quality from sparse views.

**Three components:**
1. **Difix** - Pretrained model that removes artifacts (download from HuggingFace, no training needed)
2. **Difix3D** - Training strategy that progressively refines 3D models during training
3. **Difix3D+** - Post-processing enhancement for already-rendered videos

---

## Two Usage Scenarios

### Scenario A: Enhance Existing Videos (Post-Processing)

**When:** You already have a trained 3DGS/splatfacto model and rendered video
**Time:** 5-10 minutes
**Requires:** No retraining

```bash
# Basic enhancement
python enhance_existing_video.py \
    --input your_video.mp4 \
    --output enhanced.mp4

# Better quality (with reference images from your training set)
python enhance_existing_video.py \
    --input your_video.mp4 \
    --output enhanced.mp4 \
    --use-ref \
    --ref-images /path/to/training/images
```

### Scenario B: Train From Scratch

**When:** You have sparse frames (e.g., 4 frames) and want to train a new model
**Time:** 30-40 minutes
**Requires:** COLMAP camera poses

#### Step 1: Prepare Data

```bash
# Expected structure:
data/my_scene/
├── colmap/sparse/0/  # COLMAP reconstruction
└── images/           # Your 4 frames

# Generate COLMAP if you don't have it
colmap automatic_reconstructor \
    --workspace_path data/my_scene/colmap \
    --image_path data/my_scene/images
```

#### Step 2: Train

```bash
python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir data/my_scene \
    --result_dir outputs/my_scene \
    --data_factor 4 \
    --max_steps 30000 \
    --test_every 0
```

**What happens:**
- Trains 3DGS on your sparse frames
- At steps 3000, 6000, 9000... the pretrained Difix model automatically:
  - Renders novel views
  - Fixes artifacts
  - Adds fixed views back to training
- Model progressively improves

**Results:**
- Enhanced frames: `outputs/my_scene/renders/novel/30000/Fixed/`
- Checkpoint: `outputs/my_scene/ckpts/ckpt_29999_rank0.pt`

#### Step 3: Create Video

```bash
ffmpeg -r 30 -i "outputs/my_scene/renders/novel/30000/Fixed/%04d.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 \
    outputs/my_scene/video.mp4
```

---

## Key Parameters

### For Training

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--data_factor` | Image downscale | 4 | Increase if OOM |
| `--max_steps` | Training steps | 30000 | ~30-40 min |
| `--fix_steps` | When to apply Difix | 3000, 6000... | Auto-set |
| `--test_every` | Test set sampling | 8 | Use 0 for sparse views |

### For Enhancement

| Parameter | Description | Notes |
|-----------|-------------|-------|
| `--input` | Video or image folder | Required |
| `--output` | Output path | Required |
| `--use-ref` | Use reference images | Better quality |
| `--ref-images` | Training image folder | For `--use-ref` |

---

## Quick Examples

### Example 1: Enhance Existing Video

```bash
# You have: rendered_video.mp4 and training_images/
python enhance_existing_video.py \
    --input rendered_video.mp4 \
    --output enhanced.mp4 \
    --use-ref \
    --ref-images training_images/
```

### Example 2: Train From 4 Frames

```bash
# Step 1: Setup
mkdir -p data/scene/images
cp frame_0.png data/scene/images/000000.png
cp frame_9.png data/scene/images/000009.png
cp frame_19.png data/scene/images/000019.png
cp frame_29.png data/scene/images/000029.png

# Step 2: COLMAP
colmap automatic_reconstructor \
    --workspace_path data/scene/colmap \
    --image_path data/scene/images

# Step 3: Train
python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir data/scene \
    --result_dir outputs/scene \
    --data_factor 4 \
    --max_steps 30000 \
    --test_every 0

# Step 4: Video
ffmpeg -r 30 -i "outputs/scene/renders/novel/30000/Fixed/%04d.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 \
    outputs/scene/video.mp4
```

---

## Troubleshooting

**Out of memory:**
```bash
--data_factor 8  # or higher
```

**Poor COLMAP results:**
- Use more frames (6-8 instead of 4)
- Ensure good overlap between views
- Try `ns-process-data` from Nerfstudio

**Artifacts in output:**
```bash
# Train longer
--max_steps 60000

# Or apply post-enhancement
python enhance_existing_video.py \
    --input outputs/scene/renders/novel/30000/Pred \
    --output outputs/scene/enhanced \
    --use-ref --ref-images data/scene/images
```

---

## Summary

- **Difix model:** Pretrained ✓ (just download)
- **Training:** Per-scene (30-40 min)
- **Post-processing:** Works on any model (5-10 min)
- **Best quality:** Train from scratch + post-enhancement

Choose Scenario A if you have existing models, Scenario B if training from scratch.
