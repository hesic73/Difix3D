"""
Difix3D Inference Example for Sparse Frame Video Generation

Use case: 4 frames from a 30fps video → Generate full 30fps video

This script demonstrates:
1. How to train 3DGS on sparse views
2. How to use pretrained Difix for artifact removal
3. How to render novel views for video generation
"""

import os
import torch
import numpy as np
from pathlib import Path

def step1_check_data(data_dir):
    """
    Step 1: Verify your data is in the correct COLMAP format

    Expected structure:
    data_dir/
        ├── colmap/sparse/0/
        │   ├── cameras.bin
        │   ├── images.bin
        │   └── points3D.bin
        └── images/
            ├── image_train_000000.png  # Frame 0
            ├── image_train_000009.png  # Frame 9
            ├── image_train_000019.png  # Frame 19
            └── image_train_000029.png  # Frame 29
    """
    print("=" * 80)
    print("Step 1: Checking data structure...")
    print("=" * 80)

    data_path = Path(data_dir)
    colmap_path = data_path / "colmap" / "sparse" / "0"
    images_path = data_path / "images"

    assert colmap_path.exists(), f"COLMAP data not found at {colmap_path}"
    assert images_path.exists(), f"Images not found at {images_path}"

    # Check for required COLMAP files
    required_files = ["cameras.bin", "images.bin", "points3D.bin"]
    for file in required_files:
        assert (colmap_path / file).exists(), f"Missing {file} in COLMAP data"

    # Count images
    image_files = list(images_path.glob("*.png")) + list(images_path.glob("*.jpg"))
    print(f"✓ Found {len(image_files)} images")
    print(f"✓ COLMAP data verified")
    print()


def step2_train_3dgs(data_dir, output_dir, device="cuda:0", max_steps=30000):
    """
    Step 2: Train 3DGS on sparse views with Difix3D progressive refinement

    This uses per-scene training like traditional 3DGS, but with the pretrained
    Difix model to progressively fix artifacts during training.
    """
    print("=" * 80)
    print("Step 2: Training 3DGS with Difix3D refinement...")
    print("=" * 80)
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Max steps: {max_steps}")
    print()

    from examples.gsplat.simple_trainer_difix3d import Runner, Config
    from gsplat.strategy import DefaultStrategy

    # Configure training
    cfg = Config(
        data_dir=data_dir,
        data_factor=4,  # Downsample factor (adjust based on your GPU memory)
        result_dir=output_dir,
        normalize_world_space=False,
        test_every=1,  # Every 1 image is a test image (adjust based on your data)
        max_steps=max_steps,

        # Evaluation steps
        eval_steps=[10000, 20000, 30000],
        save_steps=[10000, 20000, 30000],

        # Difix artifact fixing steps (key feature of Difix3D!)
        # At these steps, the pretrained Difix model will:
        # 1. Render novel views
        # 2. Fix artifacts using the diffusion model
        # 3. Add fixed views back to training data
        fix_steps=[3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000],

        strategy=DefaultStrategy(verbose=True),
        disable_viewer=True,
    )

    # Create runner and train
    runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)

    print("Training started...")
    print("The pretrained Difix model will be automatically used at fix_steps")
    print("to progressively improve the 3D reconstruction.")
    runner.train()

    print(f"✓ Training complete! Checkpoint saved to {output_dir}/ckpts/")
    print()


def step3_render_video(data_dir, ckpt_path, output_dir, fps=30, duration=1.0):
    """
    Step 3: Render interpolated frames for video generation

    This uses the trained 3DGS model (inference only, no training needed)
    """
    print("=" * 80)
    print("Step 3: Rendering video frames...")
    print("=" * 80)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Output: {output_dir}")
    print(f"FPS: {fps}, Duration: {duration}s")
    print()

    from examples.gsplat.simple_trainer_difix3d import Runner, Config
    from gsplat.strategy import DefaultStrategy
    from datasets.traj import generate_interpolated_path

    # Initialize runner (same as training)
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

    # Load trained checkpoint
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=runner.device)
    for k in runner.splats.keys():
        runner.splats[k].data = ckpt['splats'][k]

    # Generate interpolated camera poses
    # Your 4 training frames → interpolate to 30 frames
    train_poses = runner.parser.camtoworlds[runner.trainset.indices]
    print(f"Original training poses: {len(train_poses)}")

    # Calculate how many frames to interpolate between each pair
    total_frames = int(fps * duration)
    n_interp = (total_frames - len(train_poses)) // (len(train_poses) - 1)
    print(f"Interpolating {n_interp} frames between each pair...")
    print(f"Total output frames: ~{total_frames}")

    interpolated_poses = generate_interpolated_path(train_poses, n_interp=n_interp)

    # Render frames
    render_dir = Path(output_dir) / "renders" / "video"
    runner.render_traj(step=0, camtoworlds_all=interpolated_poses, tag="video")

    print(f"✓ Rendered {len(interpolated_poses)} frames")
    print(f"✓ Frames saved to: {render_dir}/0/Pred/")
    print()

    return render_dir / "0" / "Pred"


def step4_enhance_with_difix(input_dir, ref_images_dir, output_dir):
    """
    Step 4: (Optional) Apply Difix post-processing for final enhancement (Difix3D+)

    This uses ONLY the pretrained Difix model (no training, pure inference)
    """
    print("=" * 80)
    print("Step 4: Enhancing with pretrained Difix model (Difix3D+)...")
    print("=" * 80)
    print(f"Input: {input_dir}")
    print(f"Reference: {ref_images_dir}")
    print(f"Output: {output_dir}")
    print()

    from src.model import Difix
    from PIL import Image
    from glob import glob
    from tqdm import tqdm

    # Initialize pretrained Difix model
    print("Loading pretrained Difix model from HuggingFace...")
    model = Difix(
        pretrained_name="nvidia/difix_ref",  # Pretrained model with reference image support
        timestep=199,
        mv_unet=True,
    )
    model.set_eval()

    # Process frames
    input_images = sorted(glob(str(input_dir / "*.png")))
    ref_images = sorted(glob(str(ref_images_dir / "*.png")))

    # Use reference images cyclically if fewer than input images
    os.makedirs(output_dir, exist_ok=True)

    for i, input_path in enumerate(tqdm(input_images, desc="Enhancing frames")):
        image = Image.open(input_path).convert('RGB')

        # Select closest reference image
        ref_idx = min(i % len(ref_images), len(ref_images) - 1)
        ref_image = Image.open(ref_images[ref_idx]).convert('RGB')

        # Apply Difix (single-step diffusion)
        output_image = model.sample(
            image,
            height=576,
            width=1024,
            ref_image=ref_image,
            prompt="remove degradation"
        )

        # Save
        output_path = output_dir / f"{i:04d}.png"
        output_image.save(output_path)

    print(f"✓ Enhanced {len(input_images)} frames")
    print(f"✓ Saved to: {output_dir}")
    print()


def step5_create_video(frames_dir, output_path, fps=30):
    """
    Step 5: Combine frames into video
    """
    print("=" * 80)
    print("Step 5: Creating video...")
    print("=" * 80)
    print(f"Frames: {frames_dir}")
    print(f"Output: {output_path}")
    print(f"FPS: {fps}")
    print()

    import subprocess

    # Use ffmpeg to create video
    cmd = [
        "ffmpeg", "-y",
        "-r", str(fps),
        "-i", str(frames_dir / "%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(output_path)
    ]

    subprocess.run(cmd, check=True)
    print(f"✓ Video created: {output_path}")
    print()


# ==============================================================================
# Main workflow
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Difix3D Inference for Sparse Frame Video Generation")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to COLMAP data directory")
    parser.add_argument("--output_dir", type=str, default="outputs/my_scene", help="Output directory")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--duration", type=float, default=1.0, help="Video duration in seconds")
    parser.add_argument("--max_steps", type=int, default=30000, help="Training steps")
    parser.add_argument("--skip_training", action="store_true", help="Skip training (use existing checkpoint)")
    parser.add_argument("--ckpt_path", type=str, help="Path to checkpoint (if skip_training)")
    parser.add_argument("--enhance", action="store_true", help="Apply Difix post-processing (Difix3D+)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Check data
    step1_check_data(data_dir)

    # Step 2: Train 3DGS (or skip if checkpoint provided)
    if not args.skip_training:
        step2_train_3dgs(
            data_dir=str(data_dir),
            output_dir=str(output_dir),
            max_steps=args.max_steps
        )
        ckpt_path = output_dir / "ckpts" / f"ckpt_{args.max_steps-1}_rank0.pt"
    else:
        assert args.ckpt_path, "Must provide --ckpt_path when using --skip_training"
        ckpt_path = Path(args.ckpt_path)

    # Step 3: Render video frames
    frames_dir = step3_render_video(
        data_dir=str(data_dir),
        ckpt_path=str(ckpt_path),
        output_dir=str(output_dir),
        fps=args.fps,
        duration=args.duration
    )

    # Step 4: (Optional) Enhance with Difix
    if args.enhance:
        enhanced_dir = output_dir / "enhanced"
        step4_enhance_with_difix(
            input_dir=frames_dir,
            ref_images_dir=data_dir / "images",
            output_dir=enhanced_dir
        )
        final_frames_dir = enhanced_dir
    else:
        final_frames_dir = frames_dir

    # Step 5: Create video
    video_path = output_dir / "output_30fps.mp4"
    step5_create_video(
        frames_dir=final_frames_dir,
        output_path=video_path,
        fps=args.fps
    )

    print("=" * 80)
    print("ALL DONE!")
    print("=" * 80)
    print(f"Output video: {video_path}")
    print()
    print("Summary:")
    print("- Difix (diffusion model): PRETRAINED, used for inference only")
    print("- 3DGS reconstruction: Per-scene training required")
    print("- Difix3D: Combines both with progressive refinement")
