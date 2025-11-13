# 褪色图片修复训练指南

本指南说明如何使用DiffBIR训练一个专门用于修复褪色古画的ControlNet模型。

## 概述

与原始DiffBIR的两阶段训练不同，这个版本**只训练Stage2的ControlNet**，专门用于修复褪色图片。

### 关键特点

- ✅ 只训练ControlNet，不需要训练SwinIR
- ✅ 直接使用褪色图片作为控制条件
- ✅ 在线生成褪色效果，无需预先准备低质量图片
- ✅ 支持自定义文本prompt来增强生成质量
- ✅ 集成了add_noise.py的褪色效果

## 数据准备

### 1. 目录结构

```
your_data_root/
├── original_images/          # 1000张褪色前的高质量图片
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
└── image_prompts.csv         # 图片对应的文本描述
```

### 2. CSV文件格式

`image_prompts.csv` 应包含两列：`filename` 和 `prompt`

```csv
filename,prompt
0766.jpg,a painting of a tree with yellow flowers
0529.jpg,a painting of a pink flower with green leaves
0530.jpg,a painting of a flower on a white background
0124.jpg,a painting of a group of people in a room
...
```

**注意事项：**
- CSV文件必须包含表头行
- filename应该与图片文件名完全匹配
- prompt应该是对图片内容的准确描述

## 环境设置

### 1. 下载预训练权重

需要下载Stable Diffusion v2.1的预训练权重：

```bash
# 使用huggingface-cli下载
huggingface-cli download stabilityai/stable-diffusion-2-1-base \
    v2-1_512-ema-pruned.ckpt \
    --local-dir ./pretrained_models/sd-v2-1
```

或访问：https://huggingface.co/stabilityai/stable-diffusion-2-1-base

### 2. 安装依赖

确保已经安装了DiffBIR的所有依赖：

```bash
pip install -r requirements.txt
```

## 配置文件

编辑 `configs/train/train_fading.yaml`，更新以下路径：

```yaml
dataset:
  train:
    params:
      # 修改为你的数据路径
      image_dir: /path/to/your/original_images
      prompt_csv: /path/to/your/image_prompts.csv

train:
  # 修改为SD权重路径
  sd_path: /path/to/v2-1_512-ema-pruned.ckpt

  # 修改为输出目录
  exp_dir: experiments/fading_restoration
```

### 配置说明

#### 数据集配置

```yaml
dataset:
  train:
    params:
      out_size: 512              # 训练图片尺寸
      crop_type: center          # center/random/none
      random_fading: true        # 是否随机化褪色参数
      fading_params:
        saturation: 0.6          # 饱和度降低
        brightness: 1.4          # 亮度提升
        yellow: 0.6              # 黄色调强度
        sepia: 0.4               # 棕褐色调强度
        noise_level: 10          # 噪声强度
        darken_strength: 0.3     # 变暗强度
```

#### 训练配置

```yaml
train:
  learning_rate: 1e-4          # 学习率
  batch_size: 4                # 批次大小（根据GPU显存调整）
  num_workers: 4               # 数据加载线程数
  train_steps: 20000           # 总训练步数
  log_every: 50                # 日志记录间隔
  ckpt_every: 2000             # checkpoint保存间隔
  image_every: 500             # 图像记录间隔
  noise_aug_timestep: 0        # 条件噪声增强（0=不使用）
```

**训练步数建议：**
- 1000张图片，batch_size=4：约250 steps/epoch
- 建议训练50-100 epochs
- 总步数：12,500 - 25,000 steps

## 测试数据加载

在开始训练前，先测试数据加载是否正常：

```bash
python test_fading_dataset.py \
    --image_dir /path/to/your/original_images \
    --prompt_csv /path/to/your/image_prompts.csv \
    --num_samples 5
```

这会：
1. 验证数据加载是否正常
2. 检查图片和prompt的匹配
3. 生成一个可视化示例 `test_fading_sample.png`

## 开始训练

### 单GPU训练

```bash
python train_stage2_fading.py --config configs/train/train_fading.yaml
```

### 多GPU训练（推荐）

使用accelerate进行多GPU训练：

```bash
# 首先配置accelerate
accelerate config

# 然后启动训练
accelerate launch train_stage2_fading.py --config configs/train/train_fading.yaml
```

## 训练监控

训练过程中会在 `exp_dir` 目录下生成：

```
experiments/fading_restoration/
├── checkpoints/                # 模型checkpoint
│   ├── 0002000.pt
│   ├── 0004000.pt
│   └── ...
└── events.out.tfevents.*       # TensorBoard日志
```

### 使用TensorBoard查看训练进度

```bash
tensorboard --logdir experiments/fading_restoration
```

在浏览器打开 `http://localhost:6006` 可以看到：
- 训练损失曲线
- 生成的样本图片
- GT和LQ的对比
- 条件编码的可视化

## 从checkpoint恢复训练

如果训练中断，可以从checkpoint恢复：

```yaml
# 在train_fading.yaml中设置
train:
  resume: experiments/fading_restoration/checkpoints/0010000.pt
```

然后重新运行训练命令。

## 训练完成后

### 1. 选择最佳checkpoint

查看TensorBoard中的损失曲线和生成样本，选择最佳的checkpoint。

### 2. 使用训练好的模型进行推理

需要修改推理脚本以使用你训练的ControlNet：

```python
# 加载你训练的ControlNet权重
cldm.load_controlnet_from_ckpt(
    torch.load("experiments/fading_restoration/checkpoints/0020000.pt")
)
```

## 高级配置

### 调整褪色效果强度

如果你的数据褪色程度不同，可以调整褪色参数：

```yaml
fading_params:
  # 轻度褪色
  saturation: 0.7
  brightness: 1.2
  yellow: 0.4
  darken_strength: 0.2

  # 重度褪色
  saturation: 0.5
  brightness: 1.6
  yellow: 0.8
  darken_strength: 0.4
```

### 启用随机化增强多样性

```yaml
random_fading: true  # 在训练时随机化褪色参数
```

### 噪声增强

增加训练鲁棒性：

```yaml
train:
  noise_aug_timestep: 100  # 在条件上添加轻微噪声
```

## 常见问题

### Q: 内存不足（CUDA Out of Memory）

A: 减小batch size或降低图片尺寸：

```yaml
train:
  batch_size: 2  # 从4降到2
dataset:
  train:
    params:
      out_size: 256  # 从512降到256
```

### Q: 训练损失不下降

A: 检查以下几点：
1. 确认SD权重加载成功
2. 降低学习率：`learning_rate: 5e-5`
3. 检查数据是否正确加载
4. 尝试不使用noise augmentation

### Q: 生成的图片质量不好

A: 可能的原因：
1. 训练步数不够，继续训练
2. 褪色效果与真实数据不匹配，调整fading_params
3. prompt质量不高，改进prompt描述
4. batch size太小，增加batch size

### Q: 如何在推理时使用训练好的模型？

A: 需要创建一个推理脚本，类似于原始的inference.py，但：
1. 不使用SwinIR
2. 直接用褪色图片作为条件
3. 加载你训练的ControlNet权重

## 性能优化

### 1. 使用混合精度训练

```bash
# 在accelerate config中启用fp16
accelerate config
# 选择 fp16 或 bf16
```

### 2. 使用gradient checkpointing

已在配置中启用：
```yaml
unet_cfg:
  use_checkpoint: True
controlnet_cfg:
  use_checkpoint: True
```

### 3. 增加num_workers

```yaml
train:
  num_workers: 8  # 根据CPU核心数调整
```

## 联系和反馈

如果遇到问题，请检查：
1. 数据路径是否正确
2. 权重文件是否下载完整
3. 依赖是否正确安装
4. GPU显存是否足够

训练愉快！🎨
