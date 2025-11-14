"""
褪色图片修复推理脚本
用于使用训练好的Stage2 ControlNet模型修复褪色的古画

与原始inference.py的主要区别：
1. 不使用SwinIR（Stage1），直接用褪色图片作为ControlNet条件
2. 支持从CSV文件加载prompt，或使用默认prompt，或自动生成prompt
3. 支持对原始图片自动应用褪色效果（用于测试）
4. 简化的流程，专门针对褪色修复任务
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


def apply_fading_effect(image: np.ndarray, fading_params: dict) -> np.ndarray:
    """
    对原始图片应用褪色效果

    Args:
        image: RGB图像，范围[0, 1]，float32
        fading_params: 褪色参数字典

    Returns:
        褪色后的图像，范围[0, 1]，float32
    """
    import cv2
    from diffbir.dataset.fading_restoration import FadingRestorationDataset

    # 创建一个临时dataset对象以使用其褪色方法
    class TempDataset:
        def __init__(self, params):
            self.fading_params = params
            self.random_fading = False  # 不使用随机化

    temp_dataset = TempDataset(fading_params)

    # 将图像转换为uint8 BGR格式（OpenCV格式）
    image_uint8 = (image * 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)

    # 应用褪色效果
    from diffbir.dataset.fading_restoration import FadingRestorationDataset
    dataset_temp = FadingRestorationDataset(
        image_dir=".",  # dummy
        prompt_csv=".",  # dummy
        fading_params=fading_params,
        random_fading=False
    )

    # 使用dataset的褪色方法
    degraded_bgr = dataset_temp._synthesize_degradation(image_bgr.astype(np.float32), fading_params)

    # 转换回RGB，范围[0, 1]
    degraded_rgb = cv2.cvtColor(degraded_bgr.astype(np.uint8), cv2.COLOR_BGR2RGB)
    degraded_float = degraded_rgb.astype(np.float32) / 255.0

    return degraded_float


def auto_generate_prompt(image: np.ndarray, captioner_type: str = "simple") -> str:
    """
    自动生成图片描述prompt

    Args:
        image: RGB图像，范围[0, 1]
        captioner_type: captioner类型 ("simple" 或其他)

    Returns:
        生成的prompt
    """
    if captioner_type == "simple":
        # 简单的默认prompt
        return "Ancient Chinese painting, traditional artwork, historical image"
    else:
        # TODO: 如果需要，可以集成LLAVA或其他captioner
        return "Ancient Chinese painting, traditional artwork, historical image"


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

    # Prompt相关参数
    parser.add_argument("--prompt_csv", type=str, default=None,
                        help="包含prompts的CSV文件路径（可选）")
    parser.add_argument("--default_prompt", type=str,
                        default="Ancient Chinese painting, faded colors, historical artwork",
                        help="默认prompt（当没有CSV或CSV中找不到对应图片时使用）")
    parser.add_argument("--auto_prompt", action="store_true",
                        help="自动生成prompt（如果启用，会为每张图片生成简单的默认prompt）")

    # 褪色效果相关参数
    parser.add_argument("--apply_fading", action="store_true",
                        help="对输入图片应用褪色效果（如果你的输入是原始图片而非褪色图片）")
    parser.add_argument("--fading_strength", type=str, default="medium",
                        choices=["light", "medium", "strong"],
                        help="褪色强度（仅在--apply_fading时有效）")
    parser.add_argument("--save_faded", action="store_true",
                        help="保存褪色后的中间图片（用于对比）")

    # 采样参数
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

    # 准备褪色参数（如果需要应用褪色效果）
    fading_params = None
    if args.apply_fading:
        # 根据褪色强度选择参数
        if args.fading_strength == "light":
            fading_params = {
                'aging_type': 'darken',
                'decay_range': (0.9, 0.95),
                'darken_strength': 0.1,
                'use_brown_overlay': True,
                'overlay_opacity': 0.4,
                'noise_level': 0,
                'num_stains': 0,
                'crack_density': 0.0,
                'crack_thickness': 1,
                'crack_type': 'light',
                'saturation': 1.0,
                'brightness': 1.0,
                'yellow': 0.0,
                'sepia': 0.0,
            }
        elif args.fading_strength == "medium":
            fading_params = {
                'aging_type': 'darken',
                'decay_range': (0.85, 0.95),
                'darken_strength': 0.2,
                'use_brown_overlay': True,
                'overlay_opacity': 0.65,
                'noise_level': 5,
                'num_stains': 0,
                'crack_density': 0.0,
                'crack_thickness': 1,
                'crack_type': 'light',
                'saturation': 1.0,
                'brightness': 1.0,
                'yellow': 0.0,
                'sepia': 0.0,
            }
        else:  # strong
            fading_params = {
                'aging_type': 'darken',
                'decay_range': (0.7, 0.85),
                'darken_strength': 0.3,
                'use_brown_overlay': True,
                'overlay_opacity': 0.8,
                'noise_level': 10,
                'num_stains': 3,
                'crack_density': 0.0,
                'crack_thickness': 1,
                'crack_type': 'light',
                'saturation': 1.0,
                'brightness': 1.0,
                'yellow': 0.0,
                'sepia': 0.0,
            }
        print(f"  Will apply {args.fading_strength} fading effect to input images")

    # 如果需要保存褪色图片，创建子目录
    if args.save_faded:
        faded_output_dir = os.path.join(args.output, "faded_images")
        os.makedirs(faded_output_dir, exist_ok=True)

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

        # 加载原始图片
        original_image = load_image(str(img_path), size=args.max_size)
        h, w = original_image.shape[:2]

        # 如果需要应用褪色效果
        if args.apply_fading:
            lq_image = apply_fading_effect(original_image, fading_params)

            # 如果需要保存褪色图片
            if args.save_faded:
                faded_uint8 = (lq_image * 255).astype(np.uint8)
                faded_path = os.path.join(faded_output_dir, img_path.stem + "_faded" + img_path.suffix)
                Image.fromarray(faded_uint8).save(faded_path)
        else:
            # 输入已经是褪色图片
            lq_image = original_image

        # 获取prompt
        if args.auto_prompt:
            prompt = auto_generate_prompt(lq_image, captioner_type="simple")
        elif filename in prompts_dict:
            prompt = prompts_dict[filename]
        else:
            prompt = args.default_prompt

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
    if args.save_faded:
        print(f"Faded images saved to {faded_output_dir}")


if __name__ == "__main__":
    main()
