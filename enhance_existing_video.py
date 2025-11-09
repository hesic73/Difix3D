#!/usr/bin/env python3
"""
超简单脚本：增强已有的splatfacto渲染视频

使用方法：
    python enhance_existing_video.py --input your_video.mp4 --output enhanced.mp4

或者直接处理图像文件夹：
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
    """从视频提取帧"""
    print(f"从视频提取帧: {video_path}")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        'ffmpeg', '-i', str(video_path),
        '-vf', f'fps={fps}',
        f'{output_dir}/%04d.png'
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"✓ 提取完成，保存到: {output_dir}")

def enhance_frames(input_dir, output_dir, use_ref=False, ref_dir=None):
    """使用Difix增强图像"""
    print("\n" + "="*60)
    print("开始使用Difix增强图像...")
    print("="*60)

    # 加载Difix模型
    print("\n1. 加载预训练Difix模型...")
    from pipeline_difix import DifixPipeline

    model_name = "nvidia/difix_ref" if use_ref else "nvidia/difix"
    print(f"   模型: {model_name}")

    pipe = DifixPipeline.from_pretrained(model_name, trust_remote_code=True)
    pipe.to("cuda")
    print("   ✓ 模型加载完成")

    # 获取输入图像
    input_frames = sorted(glob(f"{input_dir}/*.png") + glob(f"{input_dir}/*.jpg"))
    if not input_frames:
        raise ValueError(f"在 {input_dir} 中没有找到图像文件")

    print(f"\n2. 找到 {len(input_frames)} 帧")

    # 准备参考图像（如果使用）
    ref_frames = None
    if use_ref and ref_dir:
        ref_frames = sorted(glob(f"{ref_dir}/*.png") + glob(f"{ref_dir}/*.jpg"))
        print(f"   使用参考图像: {len(ref_frames)} 张")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 处理每一帧
    print(f"\n3. 处理中...")
    for i, frame_path in enumerate(tqdm(input_frames, desc="增强帧")):
        # 读取输入图像
        image = Image.open(frame_path).convert('RGB')

        # 读取参考图像（如果有）
        ref_image = None
        if ref_frames:
            # 使用最接近的参考图像（循环选择）
            ref_idx = min(i % len(ref_frames), len(ref_frames) - 1)
            ref_image = Image.open(ref_frames[ref_idx]).convert('RGB')

        # Difix增强
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

        # 保存
        output_path = f"{output_dir}/{i:04d}.png"
        enhanced.save(output_path)

    print(f"\n✓ 增强完成! 保存到: {output_dir}")

def create_video(frames_dir, output_path, fps=30):
    """从帧合成视频"""
    print(f"\n4. 合成视频...")
    print(f"   输出: {output_path}")

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
    print(f"✓ 视频创建完成: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='增强已有的3DGS/splatfacto渲染视频',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 增强视频文件
  python enhance_existing_video.py --input video.mp4 --output enhanced.mp4

  # 增强图像文件夹
  python enhance_existing_video.py --input frames/ --output enhanced/

  # 使用参考图像（更好的效果）
  python enhance_existing_video.py --input video.mp4 --output enhanced.mp4 \\
      --use-ref --ref-images /path/to/training/images
        """
    )

    parser.add_argument('--input', '-i', type=str, required=True,
                        help='输入视频文件或图像文件夹')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='输出视频文件或图像文件夹')
    parser.add_argument('--fps', type=int, default=30,
                        help='视频帧率 (默认: 30)')
    parser.add_argument('--use-ref', action='store_true',
                        help='使用参考图像引导（效果更好）')
    parser.add_argument('--ref-images', type=str,
                        help='参考图像文件夹（你训练splatfacto用的原始图像）')
    parser.add_argument('--keep-temp', action='store_true',
                        help='保留临时文件')

    args = parser.parse_args()

    print("="*60)
    print("Difix3D+ 视频增强工具")
    print("="*60)
    print(f"\n输入: {args.input}")
    print(f"输出: {args.output}")

    input_path = Path(args.input)
    output_path = Path(args.output)

    # 检查输入
    if not input_path.exists():
        raise FileNotFoundError(f"输入不存在: {input_path}")

    # 判断输入是视频还是文件夹
    is_video_input = input_path.is_file() and input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']
    is_video_output = output_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']

    # 临时文件夹
    temp_dir = Path("temp_frames")
    extracted_dir = temp_dir / "extracted"
    enhanced_dir = temp_dir / "enhanced"

    try:
        # 步骤1: 提取帧（如果输入是视频）
        if is_video_input:
            extract_frames(input_path, extracted_dir, args.fps)
            input_frames_dir = extracted_dir
        else:
            input_frames_dir = input_path

        # 步骤2: 增强帧
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

        # 步骤3: 合成视频（如果输出是视频）
        if is_video_output:
            create_video(output_frames_dir, output_path, args.fps)

    finally:
        # 清理临时文件
        if not args.keep_temp and temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
            print("\n✓ 临时文件已清理")

    print("\n" + "="*60)
    print("完成!")
    print("="*60)
    print(f"\n增强后的{'视频' if is_video_output else '图像'}保存在: {output_path}")

    # 显示对比信息
    if is_video_input:
        print(f"\n原始视频: {input_path}")
        print(f"增强视频: {output_path}")
        print("\n你可以用视频播放器对比查看效果！")

if __name__ == '__main__':
    main()
