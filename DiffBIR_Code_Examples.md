# DiffBIR Stage2 条件控制 - 代码示例详解

## 一、ControlNet 完整实现代码

### 1.1 ControlNet 初始化和前向传播

**文件位置**: `/home/user/DiffBIR/diffbir/model/controlnet.py` 第50-328行

```python
class ControlNet(nn.Module):
    """
    ControlNet: 一个与UNet并行的控制网络
    输入：
    - x: 扩散模型的输入 (batch, in_channels, h, w)
    - hint: 控制条件，通常是Stage1的输出 (batch, hint_channels, h, w)
    - timesteps: 扩散时间步
    - context: 文本条件编码 (batch, seq_len, context_dim)
    
    输出：
    - 13个尺度的特征图，用于约束UNet的生成
    """

    def __init__(
        self,
        image_size,
        in_channels,           # 4 (VAE潜在维度)
        model_channels,        # 320
        hint_channels,         # 4 (Stage1输出维度)
        num_res_blocks,        # 2
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        **kwargs
    ):
        super().__init__()
        
        # 时间步嵌入
        time_embed_dim = model_channels * 4  # 320 * 4 = 1280
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        
        # 输入块：接收 (x + hint) 拼接的输入
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(
                        2,  # 2D卷积
                        in_channels + hint_channels,  # 4 + 4 = 8
                        model_channels,               # → 320
                        3,
                        padding=1
                    )
                )
            ]
        )
        
        # 零卷积：初始化为零，逐步学习
        self.zero_convs = nn.ModuleList([self.make_zero_conv(model_channels)])
        
        # 编码器层（逐级下采样）
        ch = model_channels  # 320
        for level, mult in enumerate(channel_mult):  # [1, 2, 4, 4]
            for nr in range(num_res_blocks):  # 2个ResBlock
                # ResBlock
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        use_checkpoint=use_checkpoint,
                    )
                ]
                ch = mult * model_channels
                
                # 可选的attention
                if ds in attention_resolutions:
                    layers.append(
                        SpatialTransformer(
                            ch, num_heads, dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                        )
                    )
                
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                input_block_chans.append(ch)
            
            # 下采样
            if level != len(channel_mult) - 1:
                self.input_blocks.append(...)
                self.zero_convs.append(self.make_zero_conv(ch))
        
        # 中间块
        self.middle_block = TimestepEmbedSequential(
            ResBlock(...),
            SpatialTransformer(...),
            ResBlock(...),
        )
        self.middle_block_out = self.make_zero_conv(ch)

    def make_zero_conv(self, channels):
        """
        零卷积：初始化所有权重为零
        这是ControlNet的创新点：
        - 保证初始时不影响UNet的输出
        - 通过训练逐步学习约束
        """
        return TimestepEmbedSequential(
            zero_module(conv_nd(2, channels, channels, 1, padding=0))
        )

    def forward(self, x, hint, timesteps, context, **kwargs):
        """
        x: 噪声图像 (B, 4, h, w)
        hint: 控制条件 (B, 4, h, w) - Stage1的VAE编码输出
        timesteps: 时间步 (B,)
        context: 文本编码 (B, 77, 1024)
        
        返回：13个尺度的控制信号
        """
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)
        
        # 关键步骤：将噪声和条件拼接
        x = torch.cat((x, hint), dim=1)  # (B, 8, h, w)
        
        # 前向传播通过编码器，同时提取零卷积输出
        outs = []
        h, emb, context = cast_dtype(x, emb, context)
        
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)           # 提取特征
            outs.append(zero_conv(h, emb, context))  # 零卷积提取控制信号
        
        # 中间块
        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))
        
        # 返回13个不同尺度的控制信号
        # outs[0]:  full resolution (h, w)
        # outs[1]:  1/2 resolution
        # outs[2]:  1/4 resolution
        # ...
        # outs[12]: 最小尺度 (中间层)
        return outs
```

### 1.2 ControlledUNetModel - 如何使用控制信号

```python
class ControlledUnetModel(UNetModel):
    """
    修改过的UNet，接收ControlNet的控制信号
    """
    
    def forward(
        self,
        x,
        timesteps=None,
        context=None,
        control=None,          # ControlNet的输出：13个特征图
        only_mid_control=False,
        **kwargs,
    ):
        """
        关键的控制流程：
        1. 中间块后直接加上控制信号
        2. 解码时在跳跃连接处加上多尺度控制
        """
        hs = []  # 存储跳跃连接特征
        
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)
        h, emb, context = cast_dtype(x, emb, context)
        
        # 编码器：生成多尺度特征用于跳跃连接
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        
        # 中间块
        h = self.middle_block(h, emb, context)
        
        # ====== 关键！添加ControlNet的中间层控制 ======
        if control is not None:
            h += control.pop()  # 直接在中间层添加
        
        # 解码器：逐级上采样，同时应用多尺度控制
        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                # 不使用跳跃连接控制
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                # ====== 核心！多尺度控制 ======
                # 在跳跃连接处也添加ControlNet的信号
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            
            h = module(h, emb, context)
        
        h = h.type(x.dtype)
        return self.out(h)
```

---

## 二、ControlLDM 的完整流程

**文件位置**: `/home/user/DiffBIR/diffbir/model/cldm.py`

```python
class ControlLDM(nn.Module):
    """
    ControlLDM = ControlNet + UNet + VAE + CLIP
    """
    
    def __init__(self, unet_cfg, vae_cfg, clip_cfg, controlnet_cfg, latent_scale_factor):
        super().__init__()
        self.unet = ControlledUnetModel(**unet_cfg)
        self.vae = AutoencoderKL(**vae_cfg)
        self.clip = FrozenOpenCLIPEmbedder(**clip_cfg)
        self.controlnet = ControlNet(**controlnet_cfg)
        self.scale_factor = latent_scale_factor
        
        # 默认的控制强度：每个尺度都是1.0
        self.control_scales = [1.0] * 13
    
    def vae_encode(self, image, sample=False, tiled=False, tile_size=-1):
        """
        编码图像到潜在空间
        image: (B, 3, H, W) in [-1, 1]
        返回: (B, 4, H/8, W/8) 潜在表示
        """
        if tiled:
            # 平铺编码处理大图像
            def encoder(x):
                h = VAEHook(self.vae.encoder, ...)(x)
                moments = self.vae.quant_conv(h)
                posterior = DiagonalGaussianDistribution(moments)
                return posterior
        else:
            encoder = self.vae.encode
        
        if sample:
            z = encoder(image).sample() * self.scale_factor
        else:
            # 使用posterior的mode而非采样（确定性）
            z = encoder(image).mode() * self.scale_factor
        
        return z
    
    def vae_decode(self, z, tiled=False, tile_size=-1):
        """
        解码潜在表示到图像空间
        z: (B, 4, H/8, W/8)
        返回: (B, 3, H, W)
        """
        if tiled:
            def decoder(z_tile):
                z_tile = self.vae.post_quant_conv(z_tile)
                dec = VAEHook(self.vae.decoder, ...)(z_tile)
                return dec
        else:
            decoder = self.vae.decode
        
        return decoder(z / self.scale_factor)
    
    def prepare_condition(self, cond_img, txt, tiled=False, tile_size=-1):
        """
        准备条件输入
        
        cond_img: Stage1输出的RGB图像 (B, H, W, 3) in [0, 1]
        txt: 文本提示列表
        
        返回: {
            'c_txt': CLIP编码 (B, 77, 1024),
            'c_img': VAE编码 (B, 4, H/8, W/8)
        }
        """
        return dict(
            c_txt=self.clip.encode(txt),
            c_img=self.vae_encode(
                cond_img * 2 - 1,  # [0,1] → [-1,1]
                sample=False,      # 确定性编码
                tiled=tiled,
                tile_size=tile_size,
            ),
        )
    
    def forward(self, x_noisy, t, cond):
        """
        核心推理函数
        
        x_noisy: 当前噪声 (B, 4, H, W)
        t: 时间步 (B,)
        cond: {'c_txt': (B, 77, 1024), 'c_img': (B, 4, H/8, W/8)}
        
        返回: 预测的噪声 (B, 4, H, W)
        """
        c_txt = cond["c_txt"]
        c_img = cond["c_img"]
        
        # 1. ControlNet生成13个尺度的控制信号
        control = self.controlnet(
            x=x_noisy,
            hint=c_img,          # Stage1的特征
            timesteps=t,
            context=c_txt,       # 文本条件
        )
        
        # 2. 应用控制强度缩放
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        
        # 3. UNet通过控制信号生成输出
        eps = self.unet(
            x=x_noisy,
            timesteps=t,
            context=c_txt,
            control=control,
            only_mid_control=False,  # 使用所有13个尺度
        )
        
        return eps
```

---

## 三、Pipeline 推理流程

**文件位置**: `/home/user/DiffBIR/diffbir/pipeline.py`

```python
class Pipeline:
    """
    完整的两阶段推理流程
    """
    
    def run(
        self,
        lq: np.ndarray,            # 低质量输入 (B, H, W, 3)
        steps: int = 10,           # 采样步数
        strength: float = 1.0,     # ControlNet强度
        pos_prompt: str = "...",   # 正面提示
        neg_prompt: str = "...",   # 负面提示
        cfg_scale: float = 6.0,    # CFG强度
        noise_aug: int = 0,        # 噪声增强
        **kwargs
    ) -> np.ndarray:
        """
        推理流程：
        LQ → Stage1 → 条件准备 → Stage2采样 → VAE解码 → HQ
        """
        
        # 1. 转换为张量
        lq_tensor = (
            torch.tensor(lq, dtype=torch.float32, device=self.device)
            .div(255)
            .clamp(0, 1)
            .permute(0, 3, 1, 2)
        )
        
        # ===== Stage1：质量恢复 =====
        cond_img = self.apply_cleaner(lq_tensor)  # SwinIR/BSRNet
        # cond_img: (B, 3, H, W) in [0, 1]
        
        # ===== Stage2：细节增强 =====
        sample = self.apply_cldm(
            cond_img,
            steps=steps,
            strength=strength,
            pos_prompt=pos_prompt,
            neg_prompt=neg_prompt,
            cfg_scale=cfg_scale,
            noise_aug=noise_aug,
            **kwargs
        )
        # sample: (B, 3, H, W) in [-1, 1]
        
        return sample
    
    def apply_cldm(
        self,
        cond_img: torch.Tensor,  # Stage1输出 (B, H, W, 3)
        steps: int,
        strength: float,
        pos_prompt: str,
        neg_prompt: str,
        cfg_scale: float,
        noise_aug: int,
        **kwargs
    ) -> torch.Tensor:
        """
        ControlLDM推理核心函数
        """
        bs, _, h0, w0 = cond_img.shape
        
        # ====== 步骤1：VAE编码条件 ======
        # 准备有条件的编码
        cond = self.cldm.prepare_condition(
            cond_img,
            [pos_prompt] * bs,
        )
        # cond = {'c_txt': (B, 77, 1024), 'c_img': (B, 4, h/8, w/8)}
        
        # 准备无条件的编码（用于CFG）
        uncond = self.cldm.prepare_condition(
            cond_img,
            [neg_prompt] * bs,
        )
        # uncond = {'c_txt': (B, 77, 1024), 'c_img': (B, 4, h/8, w/8)}
        # 注意：图像条件相同！
        
        h1, w1 = cond["c_img"].shape[2:]
        
        # ====== 步骤2：噪声增强（可选） ======
        if noise_aug > 0:
            # 向条件添加噪声以增加多样性
            cond["c_img"] = self.diffusion.q_sample(
                x_start=cond["c_img"],
                t=torch.full(size=(bs,), fill_value=noise_aug, device=self.device),
                noise=torch.randn_like(cond["c_img"]),
            )
            # 保持uncond的条件一致（避免CFG计算时的矛盾）
            uncond["c_img"] = cond["c_img"].detach().clone()
        
        # ====== 步骤3：设置控制强度 ======
        # 这是关键！strength参数直接影响ControlNet的约束力度
        control_scales = self.cldm.control_scales
        self.cldm.control_scales = [strength] * 13
        
        # ====== 步骤4：初始化噪声 ======
        # 如果使用noise作为起点
        x_T = torch.randn((bs, 4, h1, w1), device=self.device)
        
        # ====== 步骤5：采样 ======
        sampler = SpacedSampler(self.diffusion.betas, rescale_cfg=False)
        z = sampler.sample(
            model=self.cldm,
            device=self.device,
            steps=steps,
            x_size=(bs, 4, h1, w1),
            cond=cond,
            uncond=uncond,
            cfg_scale=cfg_scale,
            x_T=x_T,
            progress=True,
        )
        
        # 恢复control_scales
        self.cldm.control_scales = control_scales
        
        # ====== 步骤6：VAE解码 ======
        x = self.cldm.vae_decode(z)
        # x: (B, 3, H, W) in [-1, 1]
        
        return x
```

---

## 四、采样过程中的控制

**文件位置**: `/home/user/DiffBIR/diffbir/sampler/spaced_sampler.py`

```python
class SpacedSampler:
    """
    迭代去噪采样器
    """
    
    @torch.no_grad()
    def sample(
        self,
        model: ControlLDM,
        device: str,
        steps: int,
        x_size: Tuple[int],      # (B, 4, H, W)
        cond: Dict,              # {'c_txt': ..., 'c_img': ...}
        uncond: Dict,            # {'c_txt': ..., 'c_img': ...}
        cfg_scale: float = 6.0,
        x_T: torch.Tensor = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        核心采样循环：从纯噪声逐步去噪至清晰图像
        
        关键点：
        1. 条件信息（c_img）在整个采样过程中保持不变
        2. ControlNet在每一步都生成约束信号
        3. CFG用来增强文本条件的影响
        """
        
        # 1. 生成采样时间表
        self.make_schedule(steps)  # e.g., [999, 800, 600, ..., 0]
        
        # 2. 初始化噪声
        if x_T is None:
            x_T = torch.randn(x_size, device=device, dtype=torch.float32)
        
        x = x_T
        timesteps = np.flip(self.timesteps)  # 逆序：1000→0
        
        # 3. 迭代去噪
        for i, step in enumerate(tqdm(timesteps)):
            model_t = torch.full((bs,), step, device=device, dtype=torch.long)
            t = torch.full((bs,), total_steps - i - 1, device=device, dtype=torch.long)
            
            # ====== 关键：计算当前的CFG强度 ======
            cur_cfg_scale = self.get_cfg_scale(cfg_scale, step)
            
            # ====== 核心：通过模型生成预测 ======
            x = self.p_sample(
                model,        # ControlLDM
                x,            # 当前噪声 (B, 4, H, W)
                model_t,      # 当前时间步
                t,            # 指标时间步
                cond,         # 条件信息（包括c_img）
                uncond,       # 无条件信息
                cur_cfg_scale,
            )
        
        return x
    
    def p_sample(
        self,
        model: ControlLDM,
        x: torch.Tensor,
        model_t: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Dict[str, torch.Tensor],
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        单步去噪
        """
        
        # ====== 步骤1：应用条件和无条件预测（CFG） ======
        if cfg_scale == 1.0:
            # 不使用CFG
            model_output = model(x, model_t, cond)
        else:
            # CFG：使用有条件和无条件的差值来增强条件影响
            model_cond = model(x, model_t, cond)      # 完整条件
            model_uncond = model(x, model_t, uncond)  # 无条件
            
            # 关键公式：
            # output = uncond + cfg_scale * (cond - uncond)
            # 当cfg_scale > 1.0时，增强条件的影响
            model_output = model_uncond + cfg_scale * (model_cond - model_uncond)
        
        # ====== 步骤2：从预测的噪声恢复x_0 ======
        # 根据扩散过程的反向步骤预测清晰图像
        if self.parameterization == "eps":
            # ε预测：预测的是去掉的噪声
            pred_x0 = self._predict_xstart_from_eps(x, t, model_output)
        else:
            # v预测：速度预测
            pred_x0 = self._predict_xstart_from_v(x, t, model_output)
        
        # ====== 步骤3：计算下一步的均值和方差 ======
        mean, variance = self.q_posterior_mean_variance(pred_x0, x, t)
        
        # ====== 步骤4：采样下一步 ======
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        x_prev = mean + nonzero_mask * torch.sqrt(variance) * noise
        
        return x_prev
    
    def apply_model(
        self,
        model: ControlLDM,
        x: torch.Tensor,
        model_t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Dict[str, torch.Tensor],
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        模型前向传播 + CFG
        """
        if uncond is None or cfg_scale == 1.0:
            model_output = model(x, model_t, cond)
        else:
            # CFG关键步骤：计算两个预测的差值
            model_cond = model(x, model_t, cond)      # 有条件
            model_uncond = model(x, model_t, uncond)  # 无条件
            
            # 差值表示条件对生成的引导
            model_output = model_uncond + cfg_scale * (model_cond - model_uncond)
        
        return model_output
```

---

## 五、训练流程中的条件构建

**文件位置**: `/home/user/DiffBIR/train_stage2.py`

```python
def main(args):
    # ... 初始化代码 ...
    
    for batch in loader:
        # 数据来自DataLoader
        gt, lq, prompt = batch
        # gt: Ground Truth高质量图像 (B, H, W, 3)
        # lq: Low Quality输入 (B, H, W, 3)
        # prompt: 文本描述列表
        
        gt = rearrange(gt, "b h w c -> b c h w").float()
        lq = rearrange(lq, "b h w c -> b c h w").float()
        
        with torch.no_grad():
            # ====== 步骤1：编码GT（目标） ======
            # VAE编码ground truth到潜在空间
            z_0 = pure_cldm.vae_encode(gt)
            # z_0: (B, 4, H/8, W/8) - 训练的目标
            
            # ====== 步骤2：Stage1处理（冻结） ======
            # SwinIR处理低质量图像
            clean = swinir(lq)
            # clean: (B, 3, H, W) in [0, 1] - Stage1输出
            
            # ====== 步骤3：准备条件 ======
            # 将Stage1输出转换为条件向量
            cond = pure_cldm.prepare_condition(clean, prompt)
            # cond = {
            #     'c_txt': CLIP(prompt),        # (B, 77, 1024)
            #     'c_img': VAE_encode(clean)    # (B, 4, H/8, W/8)
            # }
            
            # ====== 步骤4：条件噪声增强（可选） ======
            # 在训练时可以添加噪声以增加鲁棒性
            cond_aug = copy.deepcopy(cond)
            if noise_aug_timestep > 0:
                # 向条件添加少量噪声
                cond_aug["c_img"] = diffusion.q_sample(
                    x_start=cond_aug["c_img"],
                    t=torch.randint(
                        0, noise_aug_timestep, (z_0.shape[0],), device=device
                    ),
                    noise=torch.randn_like(cond_aug["c_img"]),
                )
        
        # ====== 步骤5：随机时间步 ======
        # 在整个时间步范围内随机采样
        t = torch.randint(
            0, diffusion.num_timesteps, (z_0.shape[0],), device=device
        )
        
        # ====== 步骤6：计算扩散损失 ======
        # 这会调用 ControlLDM.forward()
        loss = diffusion.p_losses(cldm, z_0, t, cond_aug)
        # 内部流程：
        # 1. 加噪：z_t = sqrt(alpha_t) * z_0 + sqrt(1-alpha_t) * noise
        # 2. 通过cldm预测：eps_pred = cldm(z_t, t, cond)
        # 3. 计算损失：loss = ||noise - eps_pred||^2
        
        # ====== 步骤7：反向传播（只更新ControlNet） ======
        opt.zero_grad()
        accelerator.backward(loss)
        opt.step()
        
        # 其他模块保持冻结：
        # - UNet: 来自预训练的Stable Diffusion
        # - VAE: 来自预训练的Stable Diffusion
        # - CLIP: 来自OpenAI的CLIP
        # - SwinIR: 来自stage1训练
```

---

## 六、条件信息流向图

```
数据流向：

训练过程：
┌─────────────┐
│ 批量数据    │
├─────────────┤
│ gt, lq, txt │
└──────┬──────┘
       │
       ├─────→ SwinIR(lq) ──→ clean ──┐
       │                              │
       │                              ▼
       │                        prepare_condition()
       │                              │
       │          ┌────────────────────┴─────────────────┐
       │          │                                      │
       ▼          ▼                                      ▼
   VAE_encode(gt)  CLIP(txt)                    VAE_encode(clean)
      (z_0)       (c_txt)                         (c_img)
       │          │                              │
       │          └──────────────────┬───────────┘
       │                             │
       │                      cond = {c_txt, c_img}
       │                             │
       ├──────────────────────────→ diffusion.p_losses()
       │                             │
       │                    1. q_sample(z_0, t)
       │                             │
       │                    2. ControlLDM.forward()
       │                       ├─ ControlNet(z_t, c_img, t, c_txt)
       │                       └─ UNet(z_t, c_txt, control)
       │                             │
       │                    3. ||noise - eps_pred||²
       │
       └─ 优化目标：最小化预测误差


推理过程：
┌──────────────┐
│ 输入(lq)     │
└──────┬───────┘
       │
       ▼
   SwinIR(lq) ──→ clean (Stage1输出)
       │
       ▼
   prepare_condition(clean, pos_prompt, neg_prompt)
       │
       ├──→ CLIP(pos_prompt) ──→ c_txt_cond
       ├──→ CLIP(neg_prompt) ──→ c_txt_uncond
       ├──→ VAE_encode(clean) ──→ c_img_cond
       └──→ VAE_encode(clean) ──→ c_img_uncond
       │
   cond = {c_txt: c_txt_cond, c_img: c_img_cond}
   uncond = {c_txt: c_txt_uncond, c_img: c_img_uncond}
       │
   可选：noise_aug(c_img) ──→ 添加噪声增强多样性
       │
       ▼
   Sampler.sample()
   ├─ for t in [999, 800, ..., 0]:
   │  │
   │  ├─ model(x, t, cond) ──→ eps_cond (有条件)
   │  ├─ model(x, t, uncond) ──→ eps_uncond (无条件)
   │  │
   │  ├─ CFG: eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
   │  │
   │  └─ 从 eps 计算下一步 x_{t-1}
   │
   └─ 返回最终的z
       │
       ▼
   VAE_decode(z) ──→ 高质量输出
```

