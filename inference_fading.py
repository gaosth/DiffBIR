"""
褪色图片修复推理脚本
用于使用训练好的Stage2 ControlNet模型修复褪色的古画

与原始inference.py的主要区别：
1. 不使用SwinIR（Stage1），直接用褪色图片作为ControlNet条件
2. 支持从CSV文件加载prompt，或使用默认prompt
3. 简化的流程，专门针对褪色修复任务
"""

import os
from argparse import ArgumentParser
from pathlib import Path
import csv

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange

from diffbir.model import ControlLDM, Diffusion
from diffbir.utils.common import instantiate_from_config
from diffbir.sampler import SpacedSampler


def load_image(image_path: str, size: int = None) -> np.ndarray:
    """
    加载图片并转换为numpy数组

    Args:
        image_path: 图片路径
        size: 可选的resize尺寸

    Returns:
        RGB图像，范围[0, 1]，float32
    """
    image = Image.open(image_path).convert("RGB")

    if size is not None:
        # 保持长宽比resize
        w, h = image.size
        if max(w, h) > size:
            if w > h:
                new_w = size
                new_h = int(h * size / w)
            else:
                new_h = size
                new_w = int(w * size / h)
            image = image.resize((new_w, new_h), Image.LANCZOS)

    # 转换为numpy数组，范围[0, 1]
    image = np.array(image).astype(np.float32) / 255.0
    return image


def load_prompts_from_csv(csv_path: str) -> dict:
    """从CSV文件加载prompts"""
    prompts = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            prompt = row['prompt']
            prompts[filename] = prompt
    return prompts


def main():
    parser = ArgumentParser()

    # 必需参数
    parser.add_argument("--input", type=str, required=True,
                        help="输入图片目录或单个图片路径")
    parser.add_argument("--output", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="训练好的ControlNet权重路径（.pt文件）")
    parser.add_argument("--config", type=str, required=True,
                        help="训练配置文件路径（train_fading.yaml）")

    # 可选参数
    parser.add_argument("--prompt_csv", type=str, default=None,
                        help="包含prompts的CSV文件路径（可选）")
    parser.add_argument("--default_prompt", type=str,
                        default="Ancient Chinese painting, faded colors, historical artwork",
                        help="默认prompt（当没有CSV或CSV中找不到对应图片时使用）")
    parser.add_argument("--steps", type=int, default=50,
                        help="扩散采样步数（更多步数=更好质量，但更慢）")
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                        help="Classifier-free guidance scale（1.0=不使用CFG）")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="批处理大小")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="设备类型")
    parser.add_argument("--seed", type=int, default=231,
                        help="随机种子")
    parser.add_argument("--max_size", type=int, default=512,
                        help="最大图片尺寸（会保持长宽比resize）")

    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 加载配置
    print(f"Loading config from {args.config}")
    cfg = OmegaConf.load(args.config)

    # 加载prompts（如果提供了CSV文件）
    prompts_dict = {}
    if args.prompt_csv:
        print(f"Loading prompts from {args.prompt_csv}")
        prompts_dict = load_prompts_from_csv(args.prompt_csv)
        print(f"Loaded {len(prompts_dict)} prompts")

    # 加载模型
    print("Loading models...")

    # 1. 创建并加载ControlLDM
    print("  - Loading ControlLDM...")
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)

    # 加载预训练的Stable Diffusion权重
    sd_weight = torch.load(cfg.train.sd_path, map_location="cpu")
    sd_weight = sd_weight["state_dict"]
    unused, missing = cldm.load_pretrained_sd(sd_weight)
    print(f"    Loaded SD weights (unused: {len(unused)}, missing: {len(missing)})")

    # 加载训练好的ControlNet权重
    control_weight = torch.load(args.ckpt, map_location="cpu")
    cldm.load_controlnet_from_ckpt(control_weight)
    print(f"    Loaded ControlNet from {args.ckpt}")

    cldm.eval().to(args.device)

    # 2. 加载Diffusion
    print("  - Loading Diffusion...")
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    diffusion.to(args.device)

    # 3. 创建采样器
    sampler = SpacedSampler(
        diffusion.betas,
        diffusion.parameterization,
        rescale_cfg=False
    )

    print("Models loaded successfully!")

    # 获取输入图片列表
    input_path = Path(args.input)
    if input_path.is_file():
        image_files = [input_path]
    else:
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            image_files.extend(input_path.glob(ext))
            image_files.extend(input_path.glob(ext.upper()))

    print(f"\nFound {len(image_files)} images to process")

    # 处理每张图片
    for img_path in tqdm(image_files, desc="Processing images"):
        filename = img_path.name

        # 获取prompt
        if filename in prompts_dict:
            prompt = prompts_dict[filename]
        else:
            prompt = args.default_prompt

        # 加载图片
        lq_image = load_image(str(img_path), size=args.max_size)
        h, w = lq_image.shape[:2]

        # 转换为tensor并添加batch维度
        lq_tensor = torch.from_numpy(lq_image).to(args.device)
        lq_tensor = rearrange(lq_tensor, "h w c -> 1 c h w")  # [1, 3, H, W]

        with torch.no_grad():
            # 准备条件（直接使用褪色图片，范围[0,1]）
            # prepare_condition内部会自动做*2-1转换
            cond = cldm.prepare_condition(lq_tensor, [prompt])

            # 采样生成修复后的图片
            z = sampler.sample(
                model=cldm,
                device=args.device,
                steps=args.steps,
                x_size=(1, 4, h // 8, w // 8),  # latent size
                cond=cond,
                uncond=None,
                cfg_scale=args.cfg_scale,
                progress=False,
            )

            # VAE解码
            restored = cldm.vae_decode(z)  # 范围[-1, 1]
            restored = (restored + 1) / 2  # 转换到[0, 1]

            # 转换为numpy并保存
            restored = restored.squeeze(0).cpu().numpy()  # [3, H, W]
            restored = rearrange(restored, "c h w -> h w c")
            restored = np.clip(restored * 255, 0, 255).astype(np.uint8)

            # 保存结果
            output_filename = img_path.stem + "_restored" + img_path.suffix
            output_path = os.path.join(args.output, output_filename)
            Image.fromarray(restored).save(output_path)

    print(f"\nDone! Results saved to {args.output}")


if __name__ == "__main__":
    main()
