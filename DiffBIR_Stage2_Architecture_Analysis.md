# DiffBIR Stage2 生成时结构一致性保证机制研究

## 一、整体架构概述

DiffBIR 采用两阶段策略来确保生成图片的结构与原图一致：
1. **Stage1（质量恢复）**: 使用 SwinIR/BSRNet 快速恢复图像质量
2. **Stage2（细节增强）**: 使用 ControlNet-Guided Diffusion Model 生成高质量细节

Stage1 的输出直接作为 Stage2 的**结构条件**，通过 ControlNet 约束生成过程。

---

## 二、ControlNet/ControlLDM 架构详解

### 2.1 ControlLDM 整体设计

**文件**: `/home/user/DiffBIR/diffbir/model/cldm.py`

```
ControlLDM 组成：
├── UNet (ControlledUnetModel)        # 主要扩散模型
├── ControlNet                         # 控制分支（关键！）
├── VAE                               # 图像编码/解码
└── CLIP                              # 文本编码
```

### 2.2 ControlNet 核心实现

**文件**: `/home/user/DiffBIR/diffbir/model/controlnet.py`

#### 关键参数配置（来自 configs/inference/cldm.yaml）

```yaml
controlnet_cfg:
  use_checkpoint: True
  in_channels: 4              # VAE 编码维度
  hint_channels: 4            # 条件输入维度（Stage1 输出）
  model_channels: 320
  attention_resolutions: [4, 2, 1]
  num_res_blocks: 2
  channel_mult: [1, 2, 4, 4]
  use_spatial_transformer: True
  context_dim: 1024           # CLIP 文本编码维度
```

#### ControlNet 前向传播流程

```python
# 输入：
# x: 当前扩散时间步的噪声图像 (batch, 4, h, w)
# hint: Stage1 输出（VAE编码后） (batch, 4, h, w)
# timesteps: 时间步嵌入
# context: 文本提示的CLIP编码 (batch, 77, 1024)

def forward(self, x, hint, timesteps, context, **kwargs):
    # 1. 时间步嵌入
    t_emb = timestep_embedding(timesteps, self.model_channels)
    emb = self.time_embed(t_emb)
    
    # 2. 核心：将噪声和条件拼接
    x = torch.cat((x, hint), dim=1)  # (batch, 8, h, w)
    
    # 3. 编码器部分 + 零卷积提取控制信号
    outs = []
    h, emb, context = map(...cast_dtype...)
    
    for module, zero_conv in zip(self.input_blocks, self.zero_convs):
        h = module(h, emb, context)      # 提取特征
        outs.append(zero_conv(h, emb, context))  # 零卷积输出
    
    # 4. 中间块处理
    h = self.middle_block(h, emb, context)
    outs.append(self.middle_block_out(h, emb, context))
    
    # 返回13个尺度的控制信号 (对应UNet的13个输出)
    return outs  # 长度=13
```

---

## 三、Stage1 输出如何传入 Stage2

### 3.1 条件准备过程

**文件**: `/home/user/DiffBIR/diffbir/model/cldm.py` 第143-158行

```python
def prepare_condition(
    self,
    cond_img: torch.Tensor,      # Stage1输出：shape (B, H, W, 3), 值范围[0,1]
    txt: List[str],               # 文本提示列表
    tiled: bool = False,
    tile_size: int = -1,
) -> Dict[str, torch.Tensor]:
    return dict(
        # 文本条件：CLIP编码
        c_txt=self.clip.encode(txt),  # (B, 77, 1024)
        
        # 图像条件：VAE编码（关键步骤！）
        c_img=self.vae_encode(
            cond_img * 2 - 1,  # 归一化到[-1,1]
            sample=False,      # 使用posterior的mode而非采样
            tiled=tiled,
            tile_size=tile_size,
        ),  # (B, 4, h/8, w/8) - VAE编码后的潜在空间
    )
```

### 3.2 在推理中的流程

**文件**: `/home/user/DiffBIR/diffbir/pipeline.py` 第71-233行

```
推理流程：
1. 输入：LQ图像 (batch, H, W, 3)
   ↓
2. Stage1处理 (apply_cleaner) 
   → SwinIR/BSRNet → Clean图像 (batch, H, W, 3)
   ↓
3. 条件编码 (prepare_condition)
   cond_img = clean图像
   cond = dict(
       c_txt = CLIP(pos_prompt),        # 文本条件
       c_img = VAE_encode(clean)        # 图像条件
   )
   uncond = dict(
       c_txt = CLIP(neg_prompt),        # 负文本条件
       c_img = VAE_encode(clean)        # 相同的图像条件！
   )
   ↓
4. 可选：噪声增强 (Noise Augmentation)
   if noise_aug > 0:
       cond["c_img"] = q_sample(
           x_start=cond["c_img"],
           t=noise_aug_timestep,
           noise=torch.randn_like(...)
       )
   ↓
5. 设置控制强度
   control_scales = [strength] * 13  # 所有层使用相同强度
   ↓
6. Sampler采样（迭代去噪）
```

---

## 四、推理过程中的条件控制机制

### 4.1 ControlledUNetModel 的控制流程

**文件**: `/home/user/DiffBIR/diffbir/model/controlnet.py` 第16-47行

```python
class ControlledUnetModel(UNetModel):
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)
        h, emb, context = cast_dtype(x, emb, context)
        
        # 输入块：生成中间特征
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        
        # 中间块处理
        h = self.middle_block(h, emb, context)
        
        # 关键！中间块处理后加上ControlNet的控制信号
        if control is not None:
            h += control.pop()  # 添加中间层控制
        
        # 输出块：解码
        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                # 不使用跳跃连接的控制
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                # 使用ControlNet的跳跃连接控制（多尺度！）
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)
        
        h = h.type(x.dtype)
        return self.out(h)
```

### 4.2 多尺度控制信号的应用

```
ControlNet输出：13个尺度的控制信号
control = [c0, c1, c2, ..., c12]

在UNet中的应用位置：
├─ 中间块后：        control[12]（最底层特征）
├─ 输出块[0]跳跃：   control[11]
├─ 输出块[1]跳跃：   control[10]
├─ ...
└─ 输出块[11]跳跃：  control[0]（最顶层特征）

这样确保了所有尺度都被结构约束！
```

### 4.3 ControlLDM的前向传播

```python
def forward(self, x_noisy, t, cond):
    c_txt = cond["c_txt"]
    c_img = cond["c_img"]
    
    # 1. 通过ControlNet生成控制信号
    control = self.controlnet(
        x=x_noisy,        # 当前时间步的噪声
        hint=c_img,       # Stage1输出（VAE编码）
        timesteps=t,
        context=c_txt     # 文本提示
    )  # 返回13个尺度的特征
    
    # 2. 应用控制强度缩放
    control = [c * scale for c, scale in zip(control, self.control_scales)]
    
    # 3. 通过ControlledUNet预测噪声
    eps = self.unet(
        x=x_noisy,
        timesteps=t,
        context=c_txt,
        control=control,          # 传入控制信号
        only_mid_control=False    # 使用所有尺度的控制
    )
    
    return eps
```

---

## 五、关键控制参数详解

### 5.1 推理时的核心参数

**文件**: `/home/user/DiffBIR/inference.py` 第240-245行

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--strength` | float | ControlNet控制强度 | 1.0 |
| `--cfg_scale` | float | Classifier-free guidance强度 | 6.0 |
| `--noise_aug` | int | 条件噪声增强 (0-1000) | 0 |
| `--rescale_cfg` | bool | 动态调整cfg_scale | False |
| `--steps` | int | 采样步数 | 10 |

### 5.2 参数作用机制

#### 5.2.1 `strength`（ControlNet强度）

```python
# pipeline.py 第172-174行
control_scales = self.cldm.control_scales
self.cldm.control_scales = [strength] * 13  # 应用到所有13个尺度

# 在CLDM前向传播中使用
control = [c * scale for c, scale in zip(control, self.control_scales)]
```

**效果**:
- `strength=1.0`: 完全遵循Stage1的结构（默认）
- `strength<1.0`: 减弱结构约束，允许更多创意
- `strength>1.0`: 增强结构约束，但可能出现artifacts

#### 5.2.2 `cfg_scale`（Classifier-Free Guidance）

```python
# sampler.py 第153-159行
if uncond is None or cfg_scale == 1.0:
    model_output = model(x, model_t, cond)
else:
    model_cond = model(x, model_t, cond)      # 有条件预测
    model_uncond = model(x, model_t, uncond)  # 无条件预测
    # 根据两者差值增强文本条件的影响
    model_output = model_uncond + cfg_scale * (model_cond - model_uncond)
```

**效果**:
- 控制文本提示对生成的影响强度
- 不直接控制结构，但影响生成风格

#### 5.2.3 `noise_aug`（噪声增强）

```python
# pipeline.py 第160-167行
if noise_aug > 0:
    cond["c_img"] = self.diffusion.q_sample(
        x_start=cond["c_img"],
        t=torch.full(size=(bs,), fill_value=noise_aug, device=self.device),
        noise=torch.randn_like(cond["c_img"]),
    )
    uncond["c_img"] = cond["c_img"].detach().clone()
```

**效果**:
- 向Stage1输出添加噪声
- 增加生成多样性，允许更多变化
- 但可能损害结构一致性
- `noise_aug=0`: 严格遵循Stage1输出
- `noise_aug>0`: 逐渐放松对Stage1输出的约束

---

## 六、采样器（Sampler）中的控制流程

**文件**: `/home/user/DiffBIR/diffbir/sampler/spaced_sampler.py`

### 6.1 迭代采样过程

```python
def sample(
    self,
    model: ControlLDM,
    device: str,
    steps: int,
    x_size: Tuple[int],
    cond: Dict[str, torch.Tensor],        # c_img + c_txt
    uncond: Dict[str, torch.Tensor],      # c_img + c_txt
    cfg_scale: float,
    tiled: bool = False,
    ...
) -> torch.Tensor:
    # 生成采样时间表
    self.make_schedule(steps)
    
    # 初始化噪声
    x = torch.randn(x_size, device=device, dtype=torch.float32)
    
    # 逆向去噪：从t=1000→0
    for step in timesteps:
        # 1. 计算当前CFG缩放系数
        cur_cfg_scale = self.get_cfg_scale(cfg_scale, step)
        
        # 2. 通过模型（包含ControlNet）
        model_output = self.apply_model(
            model, x, model_t, cond, uncond, cur_cfg_scale
        )
        
        # 3. 预测x_0并计算下一步
        pred_x0 = self._predict_xstart_from_eps(x, t, model_output)
        mean, variance = self.q_posterior_mean_variance(pred_x0, x, t)
        
        # 4. 采样下一步（除了最后一步）
        x = mean + sqrt(variance) * noise
    
    return x  # 最终的潜在表示
```

### 6.2 Tiled采样（处理大图像）

```python
# 当启用--cldm_tiled时的处理
if tiled:
    forward = model.forward
    model.forward = make_tiled_fn(
        lambda x_tile, t, cond, hi, hi_end, wi, wi_end: (
            forward(
                x_tile,
                t,
                {
                    "c_txt": cond["c_txt"],
                    "c_img": cond["c_img"][..., hi:hi_end, wi:wi_end],  # 裁剪对应区域
                },
            )
        ),
        tile_size,
        tile_stride,
    )
```

---

## 七、训练时的条件构建

**文件**: `/home/user/DiffBIR/train_stage2.py`

### 7.1 训练数据流

```python
for batch in loader:
    gt, lq, prompt = batch  # ground truth, low quality, text prompt
    
    # 1. 编码gt（目标）
    z_0 = pure_cldm.vae_encode(gt)  # (B, 4, h/8, w/8)
    
    # 2. Stage1处理（SwinIR）
    clean = swinir(lq)  # Stage1输出，作为条件
    
    # 3. 准备条件
    cond = pure_cldm.prepare_condition(clean, prompt)
    # cond = dict(
    #     c_txt = CLIP(prompt),
    #     c_img = VAE_encode(clean)
    # )
    
    # 4. 可选：条件噪声增强
    cond_aug = copy.deepcopy(cond)
    if noise_aug_timestep > 0:
        cond_aug["c_img"] = diffusion.q_sample(
            x_start=cond_aug["c_img"],
            t=torch.randint(0, noise_aug_timestep, ...),
            noise=torch.randn_like(...)
        )
    
    # 5. 随机时间步
    t = torch.randint(0, diffusion.num_timesteps, (B,))
    
    # 6. 计算损失
    loss = diffusion.p_losses(cldm, z_0, t, cond_aug)
    
    # 7. 反向传播（只更新ControlNet！）
    opt.zero_grad()  # cldm.controlnet.parameters()
    accelerator.backward(loss)
    opt.step()
```

### 7.2 训练时的关键设计

```python
# train_stage2.py 第78行
opt = torch.optim.AdamW(
    cldm.controlnet.parameters(),  # 只训练ControlNet！
    lr=cfg.train.learning_rate
)

# 其他模块冻结
for p in swinir.parameters():
    p.requires_grad = False  # SwinIR冻结

# UNet、VAE、CLIP也在训练前冻结（在load_pretrained_sd中）
```

---

## 八、VAE编码的重要性

### 8.1 为什么使用VAE编码？

```python
# Stage1输出是RGB图像 (B, H, W, 3)
clean = swinir(lq)

# 需要VAE编码转换到潜在空间
c_img = self.vae_encode(clean * 2 - 1, sample=False)
# → (B, 4, H/8, W/8)

# 好处：
# 1. 降低维度（H×W×3 → H/8×W/8×4），节省计算
# 2. 进入扩散模型的潜在空间
# 3. 建立Stage1和Stage2的特征空间对齐
```

### 8.2 VAE编码参数

```yaml
vae_cfg:
  embed_dim: 4
  ddconfig:
    z_channels: 4
    resolution: 256
    ch: 128
    ch_mult: [1, 2, 4, 4]      # 4倍下采样
    num_res_blocks: 2
    attn_resolutions: []        # 不使用注意力
```

---

## 九、零卷积（Zero Convolution）的作用

### 9.1 设计原理

```python
# controlnet.py 第309-312行
def make_zero_conv(self, channels):
    return TimestepEmbedSequential(
        zero_module(conv_nd(self.dims, channels, channels, 1, padding=0))
    )

# util.py 中的zero_module
def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()  # 初始化为零
    return module
```

### 9.2 作用机制

```
ControlNet架构：
输入 (x + hint) → Encoder → Zero Conv → 控制信号
                                ↓
                    初始化为零，逐步学习

好处：
1. 稳定训练：初始时ControlNet不影响UNet
2. 渐进学习：从零开始逐步学习如何约束
3. 避免模式崩溃：不会突然改变UNet的行为
```

---

## 十、完整的条件控制流程图

```
输入：LQ图像 (B, H, W, 3)
  ↓
┌─────────────────────────────────────────┐
│  Stage1: SwinIR/BSRNet (冻结)           │
│  输出：Clean图像 (B, H, W, 3)            │
└──────────────┬──────────────────────────┘
               ↓
       ┌───────────────────┐
       │ VAE Encoder       │
       │ clean*2-1 → c_img │
       └────────┬──────────┘
                ↓
      (B, 4, H/8, W/8)  ← 潜在空间条件
                ↓
    ┌──────────────────────────┐
    │ ControlNet (可训练)      │
    │ 输入: (x⊕hint, t, c_txt)│
    │ 输出: 13个尺度的控制信号  │
    └────────┬─────────────────┘
             ↓
      [control * strength]
             ↓
    ┌──────────────────────────┐
    │ ControlledUNet           │
    │ 添加跳跃连接控制         │
    │ 中间层直接添加           │
    └────────┬─────────────────┘
             ↓
        预测噪声 ε
             ↓
    ┌──────────────────────────┐
    │ Sampler (采样器)         │
    │ 迭代去噪 (1000→0)       │
    │ 使用CFG引导             │
    └────────┬─────────────────┘
             ↓
      潜在表示 z
             ↓
    ┌──────────────────────────┐
    │ VAE Decoder              │
    │ z/scale_factor → 高质量   │
    └────────┬─────────────────┘
             ↓
      最终输出 (B, H, W, 3)
```

---

## 十一、关键代码位置总结

| 功能 | 文件 | 行数 |
|------|------|------|
| ControlNet架构 | `controlnet.py` | 50-328 |
| ControlledUNet | `controlnet.py` | 16-47 |
| ControlLDM | `cldm.py` | 20-211 |
| 条件准备 | `cldm.py` | 143-158 |
| Pipeline流程 | `pipeline.py` | 71-233 |
| 采样器 | `sampler/spaced_sampler.py` | 144-245 |
| 推理脚本 | `inference.py` | 55-309 |
| 训练脚本 | `train_stage2.py` | 116-226 |
| 配置文件 | `configs/train/train_stage2.yaml` | 全文 |

---

## 十二、实验建议

### 12.1 调参组合表

| 场景 | strength | cfg_scale | noise_aug | steps | 说明 |
|------|----------|-----------|-----------|-------|------|
| 严格结构 | 1.0 | 6.0 | 0 | 10 | 最严格的约束 |
| 平衡 | 0.8 | 6.0 | 5 | 15 | 建议默认值 |
| 创意生成 | 0.5 | 10.0 | 50 | 20 | 更多变化 |
| 快速推理 | 0.9 | 5.0 | 0 | 5 | 最快速度 |

### 12.2 消融实验方向

1. 关闭ControlNet的不同层（only_mid_control=True）
2. 改变VAE编码的sample参数
3. 修改control_scales的应用方式
4. 尝试不同的降噪时间表

---

## 十三、性能指标

### 13.1 内存占用

- Stage1 (SwinIR): ~2GB
- ControlLDM: ~15GB
- 完整流程: ~20GB (可通过tiled推理降低)

### 13.2 推理速度

- 512×512图像：约10-30秒 (10步)
- 支持tiled推理处理任意分辨率

