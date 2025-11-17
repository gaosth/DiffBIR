"""
成对褪色图像数据集
用于训练Stage2 ControlNet，使用真实的褪色-修复图像对

与FadingRestorationDataset的区别：
- FadingRestorationDataset: 在线生成褪色效果（synthetic degradation）
- PairedFadingDataset: 使用真实的褪色-修复图像对（real pairs）
"""

import os
import csv
from typing import Dict, Tuple, Optional, List
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PairedFadingDataset(Dataset):
    """
    成对褪色图像数据集

    目录结构:
        data_root/
            ori/          # 褪色图片（低质量，作为condition）
                img1.jpg
                img2.jpg
                ...
            fixed/        # 修复图片（高质量，作为ground truth）
                img1.jpg
                img2.jpg
                ...

    注意：ori和fixed中的文件名必须一一对应
    """

    def __init__(
        self,
        data_root: str,
        prompt_csv: str,
        out_size: int = 512,
        crop_type: str = "resize",
        ori_subdir: str = "ori",
        fixed_subdir: str = "fixed",
    ):
        """
        Args:
            data_root: 数据集根目录
            prompt_csv: CSV文件路径，包含filename和prompt列
            out_size: 输出图片尺寸
            crop_type: 裁剪类型 ('center', 'random', 'resize', 'none')
            ori_subdir: 褪色图片子目录名
            fixed_subdir: 修复图片子目录名
        """
        self.data_root = Path(data_root)
        self.out_size = out_size
        self.crop_type = crop_type
        self.ori_dir = self.data_root / ori_subdir
        self.fixed_dir = self.data_root / fixed_subdir

        # 验证目录存在
        if not self.ori_dir.exists():
            raise ValueError(f"褪色图片目录不存在: {self.ori_dir}")
        if not self.fixed_dir.exists():
            raise ValueError(f"修复图片目录不存在: {self.fixed_dir}")

        # 加载prompts
        self.prompts = self._load_prompts(prompt_csv)

        # 获取图片列表（确保ori和fixed都有对应文件）
        self.image_files = self._get_paired_images()

        if len(self.image_files) == 0:
            raise ValueError(f"未找到成对的图片！请检查ori和fixed目录中的文件名是否一致")

        print(f"PairedFadingDataset initialized:")
        print(f"  - Data root: {self.data_root}")
        print(f"  - Found {len(self.image_files)} paired images")
        print(f"  - Output size: {self.out_size}")
        print(f"  - Crop type: {self.crop_type}")

    def _load_prompts(self, csv_path: str) -> Dict[str, str]:
        """从CSV文件加载prompts"""
        prompts = {}

        if not os.path.exists(csv_path):
            print(f"Warning: CSV file not found at {csv_path}")
            print("Will use default prompt for all images")
            return prompts

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row['filename']
                prompt = row['prompt']
                # 存储时不带扩展名，方便匹配
                stem = Path(filename).stem
                prompts[stem] = prompt

        print(f"Loaded {len(prompts)} prompts from CSV")
        return prompts

    def _get_paired_images(self) -> List[str]:
        """获取成对的图片文件列表"""
        # 支持的图片格式
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

        # 获取ori目录中的文件
        ori_files = {}
        for f in self.ori_dir.iterdir():
            if f.is_file() and f.suffix.lower() in valid_extensions:
                ori_files[f.stem] = f.name

        # 获取fixed目录中的文件
        fixed_files = {}
        for f in self.fixed_dir.iterdir():
            if f.is_file() and f.suffix.lower() in valid_extensions:
                fixed_files[f.stem] = f.name

        # 找到两个目录都有的文件（按stem匹配）
        paired = []
        for stem in ori_files:
            if stem in fixed_files:
                paired.append(stem)
            else:
                print(f"Warning: {ori_files[stem]} in ori/ has no matching file in fixed/")

        for stem in fixed_files:
            if stem not in ori_files:
                print(f"Warning: {fixed_files[stem]} in fixed/ has no matching file in ori/")

        paired.sort()
        return paired

    def _load_image(self, path: Path) -> Image.Image:
        """加载图片"""
        return Image.open(path).convert("RGB")

    def _resize(self, image: Image.Image, size: int) -> np.ndarray:
        """直接resize到目标尺寸（可能改变长宽比）"""
        image = image.resize((size, size), Image.LANCZOS)
        return np.array(image)

    def _center_crop(self, image: Image.Image, size: int) -> np.ndarray:
        """中心裁剪"""
        w, h = image.size

        # 先缩放到较短边等于size
        if w < h:
            new_w = size
            new_h = int(h * size / w)
        else:
            new_h = size
            new_w = int(w * size / h)

        image = image.resize((new_w, new_h), Image.LANCZOS)

        # 中心裁剪
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        image = image.crop((left, top, left + size, top + size))

        return np.array(image)

    def _random_crop(self, image: Image.Image, size: int, seed: int = None) -> np.ndarray:
        """随机裁剪"""
        if seed is not None:
            np.random.seed(seed)

        w, h = image.size

        # 先缩放到较短边等于size
        if w < h:
            new_w = size
            new_h = int(h * size / w)
        else:
            new_h = size
            new_w = int(w * size / h)

        image = image.resize((new_w, new_h), Image.LANCZOS)

        # 随机裁剪
        left = np.random.randint(0, max(1, new_w - size + 1))
        top = np.random.randint(0, max(1, new_h - size + 1))
        image = image.crop((left, top, left + size, top + size))

        return np.array(image)

    def _process_image_pair(
        self,
        ori_image: Image.Image,
        fixed_image: Image.Image,
        crop_seed: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理图片对，确保对ori和fixed应用相同的裁剪

        Returns:
            ori_array: 褪色图片数组 [H, W, 3], uint8
            fixed_array: 修复图片数组 [H, W, 3], uint8
        """
        if self.crop_type == "resize":
            ori_array = self._resize(ori_image, self.out_size)
            fixed_array = self._resize(fixed_image, self.out_size)
        elif self.crop_type == "center":
            ori_array = self._center_crop(ori_image, self.out_size)
            fixed_array = self._center_crop(fixed_image, self.out_size)
        elif self.crop_type == "random":
            # 对于随机裁剪，需要确保ori和fixed使用相同的裁剪位置
            # 使用相同的seed来保证一致性
            if crop_seed is None:
                crop_seed = np.random.randint(0, 2**31)
            ori_array = self._random_crop(ori_image, self.out_size, seed=crop_seed)
            fixed_array = self._random_crop(fixed_image, self.out_size, seed=crop_seed)
        elif self.crop_type == "none":
            # 不裁剪，要求图片已经是out_size x out_size
            ori_array = np.array(ori_image)
            fixed_array = np.array(fixed_image)
            if ori_array.shape[:2] != (self.out_size, self.out_size):
                raise ValueError(
                    f"crop_type='none' requires images to be {self.out_size}x{self.out_size}, "
                    f"but got {ori_array.shape[:2]}"
                )
        else:
            raise ValueError(f"Unknown crop_type: {self.crop_type}")

        return ori_array, fixed_array

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回一个训练样本

        Returns:
            dict containing:
                - gt: ground truth修复图片 [3, H, W], float32, range [0, 1]
                - lq: 褪色图片（作为condition） [3, H, W], float32, range [0, 1]
                - prompt: 图片描述文本
        """
        stem = self.image_files[idx]

        # 查找实际文件名（可能有不同的扩展名）
        ori_file = None
        fixed_file = None

        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        for ext in valid_extensions:
            ori_path = self.ori_dir / (stem + ext)
            if ori_path.exists():
                ori_file = ori_path
                break

        for ext in valid_extensions:
            fixed_path = self.fixed_dir / (stem + ext)
            if fixed_path.exists():
                fixed_file = fixed_path
                break

        if ori_file is None:
            raise FileNotFoundError(f"Cannot find ori file for {stem}")
        if fixed_file is None:
            raise FileNotFoundError(f"Cannot find fixed file for {stem}")

        # 加载图片
        ori_image = self._load_image(ori_file)
        fixed_image = self._load_image(fixed_file)

        # 处理图片对
        ori_array, fixed_array = self._process_image_pair(ori_image, fixed_image)

        # 转换为float32，范围[0, 1]
        ori_float = ori_array.astype(np.float32) / 255.0
        fixed_float = fixed_array.astype(np.float32) / 255.0

        # 转换为tensor [H, W, C] -> [C, H, W]
        lq_tensor = torch.from_numpy(ori_float).permute(2, 0, 1)
        gt_tensor = torch.from_numpy(fixed_float).permute(2, 0, 1)

        # 获取prompt
        if stem in self.prompts:
            prompt = self.prompts[stem]
        else:
            # 默认prompt
            prompt = "Ancient Chinese painting, traditional artwork, historical image"

        return {
            "gt": gt_tensor,
            "lq": lq_tensor,
            "prompt": prompt
        }


if __name__ == "__main__":
    # 测试代码
    import matplotlib.pyplot as plt

    # 创建测试数据
    test_root = "/tmp/paired_fading_test"
    ori_dir = os.path.join(test_root, "ori")
    fixed_dir = os.path.join(test_root, "fixed")
    os.makedirs(ori_dir, exist_ok=True)
    os.makedirs(fixed_dir, exist_ok=True)

    # 创建测试图片
    for i in range(3):
        # 创建褪色图片（偏暗）
        ori_img = np.random.randint(50, 150, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(ori_img).save(os.path.join(ori_dir, f"test_{i}.jpg"))

        # 创建修复图片（正常颜色）
        fixed_img = np.random.randint(100, 255, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(fixed_img).save(os.path.join(fixed_dir, f"test_{i}.jpg"))

    # 创建CSV
    csv_path = os.path.join(test_root, "prompts.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'prompt'])
        writer.writerow(['test_0.jpg', 'Test image 0'])
        writer.writerow(['test_1.jpg', 'Test image 1'])
        writer.writerow(['test_2.jpg', 'Test image 2'])

    # 测试数据集
    dataset = PairedFadingDataset(
        data_root=test_root,
        prompt_csv=csv_path,
        out_size=128,
        crop_type="resize"
    )

    print(f"\nDataset length: {len(dataset)}")

    # 获取一个样本
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"GT shape: {sample['gt'].shape}, dtype: {sample['gt'].dtype}")
    print(f"LQ shape: {sample['lq'].shape}, dtype: {sample['lq'].dtype}")
    print(f"GT range: [{sample['gt'].min():.3f}, {sample['gt'].max():.3f}]")
    print(f"LQ range: [{sample['lq'].min():.3f}, {sample['lq'].max():.3f}]")
    print(f"Prompt: {sample['prompt']}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    lq_np = sample['lq'].permute(1, 2, 0).numpy()
    gt_np = sample['gt'].permute(1, 2, 0).numpy()

    axes[0].imshow(lq_np)
    axes[0].set_title(f"LQ (Faded)\n{sample['prompt']}")
    axes[0].axis('off')

    axes[1].imshow(gt_np)
    axes[1].set_title("GT (Restored)")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(test_root, "sample.png"))
    print(f"\nSample visualization saved to {test_root}/sample.png")

    # 清理测试数据
    import shutil
    shutil.rmtree(test_root)
    print("Test data cleaned up")
