#!/usr/bin/env python3
"""
Simple script to enhance existing splatfacto/3DGS rendered videos using Difix

Usage:
    python enhance_existing_video.py --input your_video.mp4 --output enhanced.mp4

Or process image folders directly:
    python enhance_existing_video.py --input rendered_frames/ --output enhanced_frames/
"""

import argparse
import os
import subprocess
from pathlib import Path
from glob import glob
from tqdm import tqdm
from PIL import Image

def extract_frames(video_path, output_dir, fps=30):
    """Extract frames from video"""
    print(f"Extracting frames from video: {video_path}")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vf', f'fps={fps}',
        f'{output_dir}/%04d.png'
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"✓ Extraction complete, saved to: {output_dir}")

def enhance_frames(input_dir, output_dir, use_ref=False, ref_dir=None):
    """Enhance images using Difix"""
    print("\n" + "="*60)
    print("Starting Difix enhancement...")
    print("="*60)

    # Load Difix model
    print("\n1. Loading pretrained Difix model...")
    from pipeline_difix import DifixPipeline

    model_name = "nvidia/difix_ref" if use_ref else "nvidia/difix"
    print(f"   Model: {model_name}")

    pipe = DifixPipeline.from_pretrained(model_name, trust_remote_code=True)
    pipe.to("cuda")
    print("   ✓ Model loaded")

    # Get input images
    input_frames = sorted(glob(f"{input_dir}/*.png") + glob(f"{input_dir}/*.jpg"))
    if not input_frames:
        raise ValueError(f"No images found in {input_dir}")

    print(f"\n2. Found {len(input_frames)} frames")

    # Prepare reference images (if using)
    ref_frames = None
    if use_ref and ref_dir:
        ref_frames = sorted(glob(f"{ref_dir}/*.png") + glob(f"{ref_dir}/*.jpg"))
        print(f"   Using reference images: {len(ref_frames)} images")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each frame
    print(f"\n3. Processing...")
    for i, frame_path in enumerate(tqdm(input_frames, desc="Enhancing frames")):
        # Load input image
        image = Image.open(frame_path).convert('RGB')

        # Load reference image (if available)
        ref_image = None
        if ref_frames:
            # Use closest reference image (cyclic selection)
            ref_idx = min(i % len(ref_frames), len(ref_frames) - 1)
            ref_image = Image.open(ref_frames[ref_idx]).convert('RGB')

        # Difix enhancement
        kwargs = {
            'prompt': 'remove degradation',
            'image': image,
            'num_inference_steps': 1,
            'timesteps': [199],
            'guidance_scale': 0.0
        }

        if ref_image is not None:
            kwargs['ref_image'] = ref_image

        enhanced = pipe(**kwargs).images[0]

        # Save
        output_path = f"{output_dir}/{i:04d}.png"
        enhanced.save(output_path)

    print(f"\n✓ Enhancement complete! Saved to: {output_dir}")

def create_video(frames_dir, output_path, fps=30):
    """Create video from frames"""
    print(f"\n4. Creating video...")
    print(f"   Output: {output_path}")

    cmd = [
        'ffmpeg', '-y',
        '-r', str(fps),
        '-i', f'{frames_dir}/%04d.png',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        str(output_path)
    ]

    subprocess.run(cmd, check=True, capture_output=True)
    print(f"✓ Video created: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Enhance existing 3DGS/splatfacto rendered videos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enhance video file
  python enhance_existing_video.py --input video.mp4 --output enhanced.mp4

  # Enhance image folder
  python enhance_existing_video.py --input frames/ --output enhanced/

  # Use reference images (better quality)
  python enhance_existing_video.py --input video.mp4 --output enhanced.mp4 \\
      --use-ref --ref-images /path/to/training/images
        """
    )

    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input video file or image folder')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output video file or image folder')
    parser.add_argument('--fps', type=int, default=30,
                        help='Video FPS (default: 30)')
    parser.add_argument('--use-ref', action='store_true',
                        help='Use reference images for guidance (better quality)')
    parser.add_argument('--ref-images', type=str,
                        help='Reference image folder (training images used for splatfacto)')
    parser.add_argument('--keep-temp', action='store_true',
                        help='Keep temporary files')

    args = parser.parse_args()

    print("="*60)
    print("Difix3D+ Video Enhancement Tool")
    print("="*60)
    print(f"\nInput: {args.input}")
    print(f"Output: {args.output}")

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Check input exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    # Determine if input/output is video or folder
    is_video_input = input_path.is_file() and input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']
    is_video_output = output_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']

    # Temporary folders
    temp_dir = Path("temp_frames")
    extracted_dir = temp_dir / "extracted"
    enhanced_dir = temp_dir / "enhanced"

    try:
        # Step 1: Extract frames (if input is video)
        if is_video_input:
            extract_frames(input_path, extracted_dir, args.fps)
            input_frames_dir = extracted_dir
        else:
            input_frames_dir = input_path

        # Step 2: Enhance frames
        if is_video_output:
            output_frames_dir = enhanced_dir
        else:
            output_frames_dir = output_path

        enhance_frames(
            input_dir=input_frames_dir,
            output_dir=output_frames_dir,
            use_ref=args.use_ref,
            ref_dir=args.ref_images
        )

        # Step 3: Create video (if output is video)
        if is_video_output:
            create_video(output_frames_dir, output_path, args.fps)

    finally:
        # Clean up temporary files
        if not args.keep_temp and temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
            print("\n✓ Temporary files cleaned up")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)
    print(f"\nEnhanced {'video' if is_video_output else 'images'} saved to: {output_path}")

    # Show comparison info
    if is_video_input:
        print(f"\nOriginal video: {input_path}")
        print(f"Enhanced video: {output_path}")
        print("\nYou can compare them in your video player!")

if __name__ == '__main__':
    main()
