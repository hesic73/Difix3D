#!/bin/bash

# Difix3D Inference Script for Sparse Frame Video Generation
# Use case: 4 frames (0, 9, 19, 29th from 30fps video) → Generate 1-second 30fps video

# ============================================================================
# CONFIGURATION
# ============================================================================
SCENE_ID="my_scene"  # Change this to your scene name
DATA_DIR="data/${SCENE_ID}"  # Your COLMAP data directory
OUTPUT_DIR="outputs/difix3d/gsplat/${SCENE_ID}"
DATA_FACTOR=1  # Downscale factor (1 = no downscale, 2 = half res, 4 = quarter res)

# ============================================================================
# STEP 1: Prepare Data
# ============================================================================
# Your data should be in COLMAP format:
# ${DATA_DIR}/
#   ├── colmap/
#   │   └── sparse/
#   │       └── 0/
#   │           ├── cameras.bin
#   │           ├── images.bin
#   │           ├── points3D.bin
#   │           └── database.db
#   └── images/
#       ├── image_train_000000.png  # Frame 0
#       ├── image_train_000009.png  # Frame 9
#       ├── image_train_000019.png  # Frame 19
#       └── image_train_000029.png  # Frame 29

echo "Step 1: Checking data structure..."
if [ ! -d "${DATA_DIR}/colmap/sparse/0" ]; then
    echo "Error: COLMAP data not found at ${DATA_DIR}/colmap/sparse/0"
    echo "Please run COLMAP first to get camera poses and sparse reconstruction"
    exit 1
fi

if [ ! -d "${DATA_DIR}/images" ]; then
    echo "Error: Images not found at ${DATA_DIR}/images"
    exit 1
fi

# ============================================================================
# STEP 2: Initial 3DGS Training (30k steps)
# ============================================================================
echo "Step 2: Training initial 3DGS model on sparse views (30k steps)..."
echo "This will take some time..."

CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA_DIR} \
    --data_factor ${DATA_FACTOR} \
    --result_dir ${OUTPUT_DIR} \
    --no-normalize-world-space \
    --test_every 1 \
    --max_steps 30000 \
    --save_steps 10000 20000 30000 \
    --eval_steps 10000 20000 30000 \
    --fix_steps 3000 6000 9000 12000 15000 18000 21000 24000 27000 30000

# Note: The fix_steps are when Difix will be applied to fix artifacts and
# generate novel views that are added back to training

# ============================================================================
# STEP 3: Load checkpoint and continue training (optional, for refinement)
# ============================================================================
# If you want to continue from a checkpoint:
# CKPT_PATH="${OUTPUT_DIR}/ckpts/ckpt_29999_rank0.pt"
#
# CUDA_VISIBLE_DEVICES=0 python examples/gsplat/simple_trainer_difix3d.py default \
#     --data_dir ${DATA_DIR} \
#     --data_factor ${DATA_FACTOR} \
#     --result_dir ${OUTPUT_DIR}_continued \
#     --no-normalize-world-space \
#     --test_every 1 \
#     --ckpt ${CKPT_PATH} \
#     --max_steps 60000

# ============================================================================
# STEP 4: Render 30fps video
# ============================================================================
echo "Step 3: Rendering 30fps video..."

# For interpolation between your 4 training views:
RENDER_DIR="${OUTPUT_DIR}/renders/video"
mkdir -p ${RENDER_DIR}

# This will render interpolated frames at 30fps
# You can modify the rendering script or use the built-in trajectory rendering

# Option A: Use the rendered frames from the fix steps
# The Difix3D training already generates fixed frames at each fix_step
# These are saved in: ${OUTPUT_DIR}/renders/novel/{step}/Fixed/

# Option B: Render new interpolated trajectory
python -c "
import torch
import numpy as np
from examples.gsplat.simple_trainer_difix3d import Runner, Config
from gsplat.strategy import DefaultStrategy
from datasets.traj import generate_interpolated_path

# Load config
cfg = Config(
    data_dir='${DATA_DIR}',
    data_factor=${DATA_FACTOR},
    result_dir='${OUTPUT_DIR}',
    normalize_world_space=False,
    test_every=1,
    strategy=DefaultStrategy(verbose=False),
    disable_viewer=True
)

# Initialize runner
runner = Runner(0, 0, 1, cfg)

# Load checkpoint
ckpt_path = '${OUTPUT_DIR}/ckpts/ckpt_29999_rank0.pt'
ckpt = torch.load(ckpt_path, map_location=runner.device)
for k in runner.splats.keys():
    runner.splats[k].data = ckpt['splats'][k]

# Get camera poses (interpolate between your 4 frames)
train_poses = runner.parser.camtoworlds[runner.trainset.indices]
interpolated_poses = generate_interpolated_path(train_poses, n_interp=7)  # 7 frames between each pair

# Render with interpolated poses
runner.render_traj(step=30000, camtoworlds_all=interpolated_poses, tag='video')
print('Rendered frames saved to: ${OUTPUT_DIR}/renders/video/30000/Pred/')
"

# ============================================================================
# STEP 5: (Optional) Apply Difix post-processing for final enhancement
# ============================================================================
echo "Step 4: Applying Difix post-processing (Difix3D+)..."

# Apply the pretrained Difix model to further enhance rendered frames
python src/inference_difix.py \
    --model_name "nvidia/difix_ref" \
    --input_image "${OUTPUT_DIR}/renders/video/30000/Pred" \
    --ref_image "${DATA_DIR}/images" \
    --prompt "remove degradation" \
    --output_dir "${OUTPUT_DIR}/final_enhanced" \
    --timestep 199

# ============================================================================
# STEP 6: Create final video
# ============================================================================
echo "Step 5: Creating final 30fps video..."

# Option 1: Use ffmpeg
ffmpeg -r 30 -i "${OUTPUT_DIR}/final_enhanced/%04d.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 \
    "${OUTPUT_DIR}/output_30fps.mp4"

# Option 2: Use the built-in video export
python src/inference_difix.py \
    --model_name "nvidia/difix_ref" \
    --input_image "${OUTPUT_DIR}/renders/video/30000/Pred" \
    --ref_image "${DATA_DIR}/images" \
    --prompt "remove degradation" \
    --output_dir "${OUTPUT_DIR}/final" \
    --timestep 199 \
    --video

echo "Done! Final video saved to: ${OUTPUT_DIR}/output_30fps.mp4"
