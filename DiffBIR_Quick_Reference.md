# DiffBIR Stage2 结构一致性控制 - 快速参考指南

## 核心概念

**DiffBIR = Stage1质量恢复 + Stage2细节增强**

- **Stage1**: SwinIR/BSRNet 快速恢复（冻结）
- **Stage2**: ControlNet引导的扩散模型（ControlLDM）

**结构保证机制**: Stage1输出 → VAE编码 → 作为ControlNet的"hint" → 约束Stage2的生成

---

## 关键文件速查表

| 功能 | 文件路径 | 关键代码 |
|------|---------|---------|
| ControlNet架构 | `/diffbir/model/controlnet.py` | L50-328 |
| ControlledUNet | `/diffbir/model/controlnet.py` | L16-47 |
| ControlLDM组合 | `/diffbir/model/cldm.py` | L20-173 |
| 推理流程 | `/diffbir/pipeline.py` | L71-233 |
| 采样器 | `/diffbir/sampler/spaced_sampler.py` | L187-245 |
| 推理入口 | `/inference.py` | L290-309 |
| 训练脚本 | `/train_stage2.py` | L116-226 |
| 配置文件 | `/configs/train/train_stage2.yaml` | 全文 |

---

## Stage1 → Stage2 数据流

```
低质量输入
    ↓
┌─────────────────────────────────────┐
│ Stage1: SwinIR/BSRNet (冻结)        │
│ 输出: RGB图像 (B, H, W, 3) [0,1]   │
└──────────────┬──────────────────────┘
               ↓
        prepare_condition()
       ┌────────┴────────┐
       ↓                  ↓
   CLIP(text)        VAE_encode(image)
   (B, 77, 1024)    (B, 4, H/8, W/8)
       ↓                  ↓
   c_txt                c_img ← 核心！这是"hint"
       │                  │
       └────────┬─────────┘
                ↓
         ControlLDM.forward()
         ├─ ControlNet(x, c_img)
         └─ UNet(x, c_txt, control)
                ↓
         预测噪声 (B, 4, H, W)
                ↓
         VAE_decode(z)
                ↓
         最终输出 (B, H, W, 3)
```

---

## ControlNet 关键设计

### 1. 输入拼接
```python
# ControlNet.forward()
x = torch.cat((x, hint), dim=1)  # (B, 8, H, W)
#                     ↑ Stage1的VAE编码输出
```

### 2. 多尺度控制（13个scale）
```python
# 返回13个不同分辨率的控制信号
control = [c0, c1, ..., c12]  
# c12: 中间层（最深层特征）
# c0-c11: 逐级上采样到full resolution
```

### 3. 零卷积（Zero Convolution）
```python
# 初始化为零，逐步学习
zero_module(conv_nd(...))  
# 保证初始时不影响UNet的输出
```

### 4. UNet中的应用
```python
# 中间层直接加上
h += control.pop()  

# 跳跃连接处加上多尺度信号
h = torch.cat([h, hs.pop() + control.pop()], dim=1)
```

---

## 关键参数速查

### 推理参数（inference.py）

| 参数 | 类型 | 默认值 | 作用 | 推荐范围 |
|------|------|--------|------|---------|
| `--strength` | float | 1.0 | ControlNet约束强度 | 0.5-1.5 |
| `--cfg_scale` | float | 6.0 | 文本引导强度 | 3.0-10.0 |
| `--noise_aug` | int | 0 | 条件噪声增强 | 0-100 |
| `--steps` | int | 10 | 采样步数 | 5-50 |
| `--rescale_cfg` | bool | False | 动态调整cfg | 默认关闭 |

### ControlNet配置（configs/inference/cldm.yaml）

```yaml
controlnet_cfg:
  in_channels: 4           # VAE潜在维度
  hint_channels: 4         # Stage1输出维度
  model_channels: 320      # 基础通道数
  channel_mult: [1, 2, 4, 4]  # 各层倍增
  use_spatial_transformer: True  # 跨注意力
  context_dim: 1024        # CLIP编码维度
```

### VAE配置

```yaml
vae_cfg:
  embed_dim: 4
  ch: 128
  ch_mult: [1, 2, 4, 4]    # 总下采样倍数: 8
  # Stage1输出 (H, W, 3) → VAE_encode → (H/8, W/8, 4)
```

---

## 参数调优指南

### 结构约束（strength）

| 值 | 效果 | 用途 |
|---|------|------|
| <0.5 | 弱约束，高创意 | 需要多样性 |
| 0.8-1.0 | 强约束，保持结构 | 推荐default |
| >1.0 | 过强约束 | 可能产生artifacts |

### 文本引导（cfg_scale）

| 值 | 效果 | 用途 |
|---|------|------|
| =1.0 | 无引导 | 基准 |
| 3-6 | 适度引导 | 推荐 |
| >10 | 强引导 | 特殊风格 |

### 条件多样性（noise_aug）

| 值 | 效果 | 用途 |
|---|------|------|
| =0 | 完全遵循Stage1 | 最稳定 |
| 1-10 | 微小变化 | 多样性+稳定性 |
| >50 | 大幅变化 | 创意生成 |

---

## 推理命令示例

### 基础推理（最严格的结构约束）
```bash
python inference.py \
  --task sr \
  --version v2.1 \
  --input input_dir \
  --output output_dir \
  --steps 10 \
  --strength 1.0 \
  --cfg_scale 6.0 \
  --noise_aug 0
```

### 平衡推理（推荐）
```bash
python inference.py \
  --task sr \
  --version v2.1 \
  --input input_dir \
  --output output_dir \
  --steps 15 \
  --strength 0.8 \
  --cfg_scale 6.0 \
  --noise_aug 5
```

### 创意推理（允许更多变化）
```bash
python inference.py \
  --task sr \
  --version v2.1 \
  --input input_dir \
  --output output_dir \
  --steps 20 \
  --strength 0.5 \
  --cfg_scale 10.0 \
  --noise_aug 50
```

---

## 训练参数（train_stage2.py）

```python
# 关键配置
learning_rate: 1e-4        # ControlNet学习率
batch_size: 256            # 根据GPU调整
train_steps: 30000         # 或80000
noise_aug_timestep: 0      # 训练时不用噪声增强

# 优化目标
optimizer: AdamW
parameters: cldm.controlnet.parameters()  # 只训练ControlNet！

# 冻结模块
- UNet: 来自Stable Diffusion预训练
- VAE: 来自Stable Diffusion预训练
- CLIP: 来自OpenAI CLIP
- SwinIR: 来自stage1训练
```

---

## 条件准备过程（核心！）

```python
# 步骤1: Stage1输出
clean = swinir(lq)  # (B, H, W, 3)

# 步骤2: 准备条件
cond = cldm.prepare_condition(clean, prompt)
# 等价于：
cond = {
    'c_txt': clip.encode(prompt),           # (B, 77, 1024)
    'c_img': vae.encode(clean*2-1)          # (B, 4, H/8, W/8)
}

# 步骤3: 准备无条件（用于CFG）
uncond = {
    'c_txt': clip.encode(neg_prompt),       # (B, 77, 1024)
    'c_img': vae.encode(clean*2-1)          # 相同的图像条件！
}

# 步骤4: 可选噪声增强
if noise_aug > 0:
    cond['c_img'] = diffusion.q_sample(cond['c_img'], t=noise_aug)
    uncond['c_img'] = cond['c_img'].clone()
```

---

## ControlLDM 前向传播

```python
def forward(self, x_noisy, t, cond):
    """
    x_noisy: (B, 4, H, W) - 当前去噪步骤
    t: (B,) - 时间步
    cond: {'c_txt': (B, 77, 1024), 'c_img': (B, 4, H/8, W/8)}
    """
    
    # 1. ControlNet生成13个控制信号
    control = self.controlnet(
        x=x_noisy,
        hint=cond['c_img'],      # ← Stage1特征
        timesteps=t,
        context=cond['c_txt']    # ← 文本提示
    )  # 返回13个尺度的特征
    
    # 2. 应用强度缩放
    control = [c * scale for c, scale in zip(control, self.control_scales)]
    
    # 3. UNet with ControlNet
    eps = self.unet(
        x=x_noisy,
        timesteps=t,
        context=cond['c_txt'],
        control=control,         # ← 关键！
        only_mid_control=False
    )
    
    return eps  # 预测的噪声
```

---

## 采样循环（Sampler）

```python
# 简化的采样循环
for step in timesteps:  # [999, 800, ..., 0]
    
    # CFG: 计算有条件和无条件的差值
    eps_cond = model(x, t, cond)           # 有condition
    eps_uncond = model(x, t, uncond)       # 无condition
    eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
    
    # 根据预测的噪声更新x
    x = update_step(x, eps, t)

return x  # 最终的潜在表示
```

---

## 常见问题排查

### Q1: 生成的图像与Stage1输出偏差很大
**A**: 
- 降低 `--strength` (尝试 0.9 或 0.8)
- 降低 `--noise_aug` (尝试 0)
- 增加 `--steps` (尝试 20)

### Q2: 生成图像有闪烁/不稳定
**A**:
- 增加 `--strength` (尝试 1.2)
- 降低 `--cfg_scale` (尝试 4.0)
- 使用更多步数

### Q3: 推理速度很慢
**A**:
- 减少 `--steps` (尝试 5-8)
- 启用 `--cldm_tiled`
- 使用 `--vae_decoder_tiled`

### Q4: 文本提示没有效果
**A**:
- 增加 `--cfg_scale` (尝试 8-10)
- 使用更具体的提示词
- 检查captioner是否正确

---

## 性能指标

### 内存消耗
- SwinIR: ~2GB
- ControlLDM: ~15GB
- 总计: ~18GB (可用tiled推理降低)

### 推理速度（512×512）
- 5步: ~8秒
- 10步: ~15秒
- 20步: ~30秒

### Tiled推理开销
- tiled_size=512: 增加约20% 时间
- 内存节省: 可处理任意分辨率

---

## 测试命令集

### 测试结构约束
```bash
# 最强约束
python inference.py --input in --output out --strength 1.0 --noise_aug 0 --steps 10

# 中等约束
python inference.py --input in --output out --strength 0.8 --noise_aug 5 --steps 15

# 弱约束
python inference.py --input in --output out --strength 0.5 --noise_aug 50 --steps 20
```

### 测试CFG强度
```bash
# 弱引导
python inference.py --input in --output out --cfg_scale 3.0

# 中等引导（默认）
python inference.py --input in --output out --cfg_scale 6.0

# 强引导
python inference.py --input in --output out --cfg_scale 10.0
```

### 大图像测试
```bash
python inference.py \
  --input in \
  --output out \
  --cleaner_tiled \
  --vae_encoder_tiled \
  --vae_decoder_tiled \
  --cldm_tiled \
  --cldm_tile_size 512
```

---

## 总结

**核心原理**：
1. Stage1快速恢复基本结构
2. VAE编码Stage1输出 → c_img
3. ControlNet以c_img为"约束信号"生成13个多尺度特征
4. UNet在ControlNet约束下生成高质量细节
5. VAE解码得到最终输出

**控制参数三角形**：
```
       结构保真度
          /\
         /  \
        /    \
   strength  cfg_scale
       |      |
       |      |
    噪声增强/多样性
```

**调参金律**：
- 结构约束 → 调 `strength`
- 文本引导 → 调 `cfg_scale`
- 多样性 → 调 `noise_aug`
- 推理速度 → 调 `steps`

