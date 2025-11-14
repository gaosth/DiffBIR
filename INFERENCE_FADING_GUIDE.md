# 褪色图片修复推理指南

使用训练好的Stage2 ControlNet模型修复褪色的古画。

## 前提条件

1. 已完成训练，获得ControlNet权重文件（例如：`experiments/fading_restoration/checkpoints/0020000.pt`）
2. 有需要修复的褪色图片
3. （可选）准备好图片对应的prompt CSV文件

## 基本使用

### 最简单的用法

```bash
python inference_fading.py \
    --input /path/to/faded_images \
    --output /path/to/output \
    --ckpt experiments/fading_restoration/checkpoints/0020000.pt \
    --config configs/train/train_fading.yaml
```

这会使用默认prompt处理输入目录中的所有图片。

### 使用CSV文件指定prompt

```bash
python inference_fading.py \
    --input /path/to/faded_images \
    --output /path/to/output \
    --ckpt experiments/fading_restoration/checkpoints/0020000.pt \
    --config configs/train/train_fading.yaml \
    --prompt_csv /path/to/image_prompts.csv
```

CSV文件格式：
```csv
filename,prompt
image1.jpg,古代山水画，墨色浓郁
image2.jpg,清代人物画，工笔细腻
```

### 处理单张图片

```bash
python inference_fading.py \
    --input /path/to/single_image.jpg \
    --output /path/to/output \
    --ckpt experiments/fading_restoration/checkpoints/0020000.pt \
    --config configs/train/train_fading.yaml \
    --default_prompt "Ancient landscape painting"
```

## 参数说明

### 必需参数

| 参数 | 说明 |
|------|------|
| `--input` | 输入图片路径（可以是文件或目录） |
| `--output` | 输出目录 |
| `--ckpt` | 训练好的ControlNet权重路径 |
| `--config` | 训练时使用的配置文件 |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--prompt_csv` | None | 包含prompts的CSV文件 |
| `--default_prompt` | "Ancient Chinese painting..." | 默认prompt |
| `--steps` | 50 | 扩散采样步数（10-100） |
| `--cfg_scale` | 1.0 | CFG引导强度（1.0-10.0） |
| `--batch_size` | 1 | 批处理大小 |
| `--device` | cuda | 设备类型（cuda/cpu） |
| `--seed` | 231 | 随机种子 |
| `--max_size` | 512 | 最大图片尺寸 |

## 高级设置

### 调整生成质量

**更高质量（更慢）**：
```bash
python inference_fading.py \
    --input /path/to/images \
    --output /path/to/output \
    --ckpt path/to/checkpoint.pt \
    --config configs/train/train_fading.yaml \
    --steps 100 \
    --cfg_scale 2.0
```

**更快速度（质量略低）**：
```bash
python inference_fading.py \
    --input /path/to/images \
    --output /path/to/output \
    --ckpt path/to/checkpoint.pt \
    --config configs/train/train_fading.yaml \
    --steps 20 \
    --cfg_scale 1.0
```

### 处理大图片

如果GPU内存不足，减小 `--max_size`：

```bash
python inference_fading.py \
    --input /path/to/large_images \
    --output /path/to/output \
    --ckpt path/to/checkpoint.pt \
    --config configs/train/train_fading.yaml \
    --max_size 384  # 减小尺寸
```

## 选择最佳Checkpoint

训练过程中会保存多个checkpoint，例如：
- `0002000.pt` - 训练2000步
- `0010000.pt` - 训练10000步
- `0020000.pt` - 训练20000步

**如何选择**：
1. 查看TensorBoard的loss曲线，选择loss稳定的checkpoint
2. 使用几个不同的checkpoint测试同一张图片，目视比较效果
3. 一般来说，训练后期的checkpoint效果更好，但要避免过拟合

## 示例工作流程

### 1. 准备测试集

```bash
# 创建测试目录
mkdir -p test_images
mkdir -p test_output

# 复制一些褪色图片到test_images/
cp /path/to/faded/*.jpg test_images/
```

### 2. 测试多个checkpoints

```bash
# 测试checkpoint 10000
python inference_fading.py \
    --input test_images \
    --output test_output/ckpt_10000 \
    --ckpt experiments/fading_restoration/checkpoints/0010000.pt \
    --config configs/train/train_fading.yaml

# 测试checkpoint 20000
python inference_fading.py \
    --input test_images \
    --output test_output/ckpt_20000 \
    --ckpt experiments/fading_restoration/checkpoints/0020000.pt \
    --config configs/train/train_fading.yaml
```

### 3. 比较结果

目视比较 `test_output/ckpt_10000/` 和 `test_output/ckpt_20000/` 中的结果，选择效果最好的checkpoint。

### 4. 批量处理

使用选定的checkpoint处理所有图片：

```bash
python inference_fading.py \
    --input /path/to/all_faded_images \
    --output /path/to/final_results \
    --ckpt experiments/fading_restoration/checkpoints/0020000.pt \
    --config configs/train/train_fading.yaml \
    --prompt_csv /path/to/prompts.csv \
    --steps 50
```

## 常见问题

### Q: 输出图片还是很暗/很淡？

**A**: 可能的原因和解决方法：
1. **Checkpoint未训练充分** → 继续训练或使用后期的checkpoint
2. **训练时归一化有问题** → 确保使用修复后的训练脚本（已修复双重归一化bug）
3. **采样步数太少** → 增加 `--steps` 到 50-100

### Q: 生成的图片与原图差异太大？

**A**: 调整参数：
1. **降低创造性** → 减小 `--cfg_scale` 到 1.0
2. **增加步数** → 提高 `--steps` 到 100
3. **使用更准确的prompt** → 提供详细的图片描述

### Q: GPU内存不足？

**A**: 减小图片尺寸：
```bash
--max_size 384  # 或更小
```

### Q: 处理速度太慢？

**A**: 减少采样步数：
```bash
--steps 20  # 默认是50
```

### Q: 想要确定性结果？

**A**: 固定随机种子：
```bash
--seed 42  # 使用固定值
```

## 输出说明

脚本会为每张输入图片生成一张修复后的图片：

```
输入: image1.jpg
输出: image1_restored.jpg

输入: painting.png
输出: painting_restored.png
```

输出图片会保存在指定的 `--output` 目录中。

## 技术细节

### 推理流程

1. **加载模型**：
   - 加载预训练的Stable Diffusion v2.1权重
   - 加载训练好的ControlNet权重
   - 创建扩散采样器

2. **处理图片**：
   - 读取褪色图片，范围 [0, 1]
   - 直接用褪色图片作为ControlNet条件（不经过SwinIR）
   - VAE编码条件图片到latent space

3. **扩散采样**：
   - 从随机噪声开始（或从条件latent开始）
   - 通过多步去噪生成清晰的latent
   - ControlNet提供结构约束

4. **解码输出**：
   - VAE解码latent到像素空间
   - 转换范围 [-1, 1] → [0, 255]
   - 保存为图片

### 与训练的一致性

推理脚本与训练脚本保持一致：
- ✅ 直接使用输入图片作为条件（不用SwinIR）
- ✅ 使用相同的VAE编码方式
- ✅ 使用相同的归一化范围
- ✅ 使用相同的prompt格式

## 性能参考

在NVIDIA RTX 3090上：
- 512x512图片，50步采样：~5秒/张
- 512x512图片，100步采样：~10秒/张
- 384x384图片，50步采样：~3秒/张

## 进一步优化

如果需要更快的推理速度，可以考虑：
1. 使用 DDIM sampler（更少步数达到相似质量）
2. 量化模型（fp16 → int8）
3. 使用TensorRT加速
4. 批处理多张图片

---

**祝使用愉快！如有问题，请参考训练时的配置和日志。**
