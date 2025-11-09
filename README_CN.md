# Difix3D 详解 (中文说明)

## Paper 核心思想

这篇paper提出了用**单步扩散模型**来提升3D重建质量的方法。主要包含三个层次：

### 1. **Difix** - 单步扩散模型（预训练，可直接使用）

**问题：** 3DGS/NeRF渲染的图像经常有artifacts（模糊、floaters、缺失细节等）

**解决：** 训练一个单步扩散模型，输入有artifacts的图像，输出干净的图像

**关键点：**
- ✅ **已经预训练好**，可以直接从HuggingFace下载使用
- ⚡ **单步推理**，速度很快（不像传统diffusion需要50-1000步）
- 🎯 专门针对3D重建的artifacts训练
- 📷 支持参考图像引导（`nvidia/difix_ref`版本）

```
有artifacts的渲染图 → [Difix] → 干净的图像
```

### 2. **Difix3D** - 渐进式3D更新（训练时使用）

**问题：** 稀疏视角（比如你的4帧）训练3DGS，novel view质量差

**解决：** 在训练过程中，定期做以下操作：
1. 渲染一些novel views（新视角）
2. 用Difix修复这些渲染的artifacts
3. 把修复后的图像加回训练集
4. 继续训练3DGS

**效果：** 通过这种"self-improvement"循环，3DGS模型越来越好

```
训练流程：
初始4帧 → 训练3DGS (3k步) → 渲染novel views → Difix修复 → 加入训练
                  ↑                                               ↓
                  └─────────────── 继续训练 ←─────────────────────┘
                  (重复多次，每次3DGS都变得更好)
```

### 3. **Difix3D+** - 实时后处理（渲染时使用）

**问题：** 即使用Difix3D训练，渲染出来的图像还是可能有些模糊或缺失细节

**解决：** 在最终渲染视频时，对每一帧应用Difix后处理

**关键点：**
- ✅ **不需要重新训练3DGS**
- ✅ 可以用在任何已有的3DGS/NeRF模型上
- ⚡ 实时渲染（单步diffusion很快）

```
你的情况：
已训练的splatfacto模型 → 渲染视频帧 → [Difix后处理] → 增强后的视频
```

---

## 你的情况：已经有训练好的splatfacto模型

### 回答你的问题：

**Q1: 需要重新训练吗？**
❌ **不需要！** 你可以直接使用 **Difix3D+** 方法（后处理）

**Q2: 怎么做？**
只需要对你已经渲染好的视频帧，用Difix模型进行增强即可。

---

## 方案A：直接后处理已有视频（推荐，最简单）

### 步骤1：准备你的视频帧

```bash
# 把你的视频分解成帧（如果还没有的话）
mkdir rendered_frames
ffmpeg -i your_video.mp4 rendered_frames/%04d.png
```

### 步骤2：用Difix增强

```bash
# 使用预训练的Difix模型进行后处理
python src/inference_difix.py \
    --model_name "nvidia/difix" \
    --input_image "rendered_frames" \
    --prompt "remove degradation" \
    --output_dir "enhanced_frames" \
    --timestep 199
```

**如果你有参考图像（training views），效果会更好：**

```bash
python src/inference_difix.py \
    --model_name "nvidia/difix_ref" \
    --input_image "rendered_frames" \
    --ref_image "path/to/training/images" \  # 你训练3DGS用的原始图像
    --prompt "remove degradation" \
    --output_dir "enhanced_frames" \
    --timestep 199
```

### 步骤3：合成视频

```bash
ffmpeg -r 30 -i enhanced_frames/%04d.png -c:v libx264 -crf 18 enhanced_video.mp4
```

### 完整示例脚本：

```python
from pipeline_difix import DifixPipeline
from PIL import Image
from glob import glob
from tqdm import tqdm
import os

# 1. 加载预训练Difix模型
pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
pipe.to("cuda")

# 2. 处理每一帧
input_frames = sorted(glob("rendered_frames/*.png"))
output_dir = "enhanced_frames"
os.makedirs(output_dir, exist_ok=True)

for i, frame_path in enumerate(tqdm(input_frames)):
    # 读取帧
    image = Image.open(frame_path).convert('RGB')

    # Difix增强（单步推理，很快！）
    enhanced = pipe(
        prompt="remove degradation",
        image=image,
        num_inference_steps=1,
        timesteps=[199],
        guidance_scale=0.0
    ).images[0]

    # 保存
    enhanced.save(f"{output_dir}/{i:04d}.png")

print("Done! 增强后的帧保存在:", output_dir)
```

---

## 方案B：从头用Difix3D重新训练（可选，效果可能更好）

如果你想尝试paper里的完整方法（训练过程中就用Difix渐进改进），可以：

### 使用gsplat版本：

```bash
SCENE_ID="my_scene"
DATA_DIR="path/to/your/colmap/data"
OUTPUT_DIR="outputs/difix3d/${SCENE_ID}"

python examples/gsplat/simple_trainer_difix3d.py default \
    --data_dir ${DATA_DIR} \
    --data_factor 4 \
    --result_dir ${OUTPUT_DIR} \
    --max_steps 30000 \
    --fix_steps 3000 6000 9000 12000 15000 18000 21000 24000 27000 30000
```

### 使用nerfstudio版本：

```bash
# 首先按照README安装Difix3D的nerfstudio插件
cd examples/nerfstudio
pip install -e .
cd ../..

# 然后训练（需要从checkpoint加载，或者从头训）
ns-train difix3d \
    --data ${DATA} \
    --pipeline.model.appearance-embed-dim 0 \
    --pipeline.model.camera-optimizer.mode off \
    --output_dir ${OUTPUT_DIR} \
    --max_num_iterations 30000 \
    nerfstudio-data --downscale_factor 4
```

**注意：** 这个方案需要重新训练，但会在训练过程中就利用Difix改进模型。

---

## 对比：方案A vs 方案B

| 特性 | 方案A (后处理) | 方案B (重新训练) |
|-----|--------------|----------------|
| 需要重新训练？ | ❌ 不需要 | ✅ 需要 (~30min) |
| 能用在已有模型？ | ✅ 可以 | ❌ 需要从头来 |
| 处理速度 | ⚡ 快（~1秒/帧） | 🐢 慢（需要完整训练） |
| 效果 | ✨ 好（去除artifacts） | ✨✨ 更好（3D几何也改进） |
| 推荐场景 | 已有模型，想快速增强 | 从头开始，追求最佳质量 |

---

## 推荐流程（针对你的情况）

### 最简单的方法（5分钟搞定）：

```bash
# 1. 安装依赖
cd /home/user/Difix3D
pip install -r requirements.txt

# 2. 增强你的视频
python -c "
from pipeline_difix import DifixPipeline
from PIL import Image
from glob import glob
from tqdm import tqdm
import os

pipe = DifixPipeline.from_pretrained('nvidia/difix', trust_remote_code=True)
pipe.to('cuda')

# 替换成你的视频帧路径
input_frames = sorted(glob('rendered_frames/*.png'))
os.makedirs('enhanced', exist_ok=True)

for i, path in enumerate(tqdm(input_frames)):
    img = Image.open(path).convert('RGB')
    out = pipe('remove degradation', image=img,
               num_inference_steps=1, timesteps=[199],
               guidance_scale=0.0).images[0]
    out.save(f'enhanced/{i:04d}.png')
"

# 3. 合成视频
ffmpeg -r 30 -i enhanced/%04d.png -c:v libx264 -crf 18 enhanced_video.mp4
```

---

## 技术细节

### Difix模型架构
- 基于SD-Turbo (单步蒸馏的Stable Diffusion)
- 训练数据：从各种3D重建方法（3DGS, NeRF等）生成的有artifacts图像 + 对应GT
- 输入：有artifacts的渲染图（可选：参考图像）
- 输出：去除artifacts的干净图像
- 推理时间：~0.5-1秒/张 (单步！)

### 为什么单步就够了？
- 传统diffusion从纯噪声开始，需要多步去噪
- Difix从"接近正确但有artifacts"的图像开始，只需要小幅修正
- 通过蒸馏技术，把多步过程压缩到一步

### Reference Image的作用
- 帮助Difix理解场景的真实纹理和细节
- 选择离当前渲染视角最近的training view作为reference
- 使用`nvidia/difix_ref`模型时可用

---

## 总结

### 针对你的情况（已有splatfacto模型和渲染视频）：

1. ✅ **推荐方案A（后处理）**：
   - 不需要重新训练
   - 直接用Difix增强你的视频帧
   - 5-10分钟搞定
   - 效果：去除模糊、floaters、提升细节

2. 可选方案B（重新训练）：
   - 如果你想追求最佳效果
   - 从头用Difix3D训练
   - 需要30分钟-1小时
   - 效果：3D几何也会改进，novel view质量更好

### Paper的核心贡献：

1. **Difix模型**：单步扩散模型，专门处理3D重建artifacts
2. **Difix3D训练策略**：训练时渐进式自我改进
3. **Difix3D+应用**：可以作为任何3D方法的后处理插件

你的情况最适合直接用**Difix3D+**（后处理），不需要重新训练！
