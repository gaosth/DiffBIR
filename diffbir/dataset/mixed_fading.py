"""
混合褪色图像数据集
支持同时从unpaired（在线生成褪色）和paired（真实图像对）数据集中采样

使用场景：
- 有少量真实的褪色-修复图像对（paired）
- 有大量原始图像可以用于在线生成褪色效果（unpaired）
- 希望结合两者的优势进行训练
"""

import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Optional

from .fading_restoration import FadingRestorationDataset
from .paired_fading import PairedFadingDataset


class MixedFadingDataset(Dataset):
    """
    混合褪色数据集

    结合unpaired（在线生成褪色）和paired（真实图像对）两种数据进行训练
    """

    def __init__(
        self,
        # Unpaired数据集参数
        unpaired_image_dir: str = None,
        unpaired_prompt_csv: str = None,
        unpaired_out_size: int = 512,
        unpaired_crop_type: str = "resize",
        # 褪色参数（用于unpaired）- 新格式
        fading_params: dict = None,
        random_fading: bool = True,
        # 褪色参数（用于unpaired）- 旧格式（兼容）
        aging_type: str = None,
        decay_range: tuple = None,
        darken_strength: float = None,
        use_brown_overlay: bool = None,
        overlay_opacity: float = None,
        # Paired数据集参数
        paired_data_root: str = None,
        paired_prompt_csv: str = None,
        paired_out_size: int = 512,
        paired_crop_type: str = "resize",
        paired_ori_subdir: str = "ori",
        paired_fixed_subdir: str = "fixed",
        # 混合参数
        paired_ratio: float = 0.5,  # paired数据的采样比例
        total_samples: int = None,  # 每个epoch的总样本数，None则自动计算
    ):
        """
        Args:
            unpaired_*: FadingRestorationDataset的参数
            paired_*: PairedFadingDataset的参数
            fading_params: 褪色参数字典，传递给FadingRestorationDataset
            random_fading: 是否随机生成褪色参数
            paired_ratio: paired数据的采样比例，范围[0, 1]
                - 0.0: 只使用unpaired数据
                - 1.0: 只使用paired数据
                - 0.5: 50% unpaired + 50% paired
            total_samples: 每个epoch的总样本数
                - None: 自动计算为两个数据集长度之和
                - 具体数值: 使用指定的样本数
        """
        self.paired_ratio = paired_ratio

        # 处理褪色参数：支持新旧两种格式
        # 如果单独的参数被设置，将它们合并到fading_params中
        if fading_params is None:
            fading_params = {}

        # 旧格式参数覆盖（如果设置了的话）
        if aging_type is not None:
            fading_params['aging_type'] = aging_type
        if decay_range is not None:
            fading_params['decay_range'] = decay_range
        if darken_strength is not None:
            fading_params['darken_strength'] = darken_strength
        if use_brown_overlay is not None:
            fading_params['use_brown_overlay'] = use_brown_overlay
        if overlay_opacity is not None:
            fading_params['overlay_opacity'] = overlay_opacity

        # 初始化unpaired数据集
        self.unpaired_dataset = None
        if unpaired_image_dir is not None and paired_ratio < 1.0:
            self.unpaired_dataset = FadingRestorationDataset(
                image_dir=unpaired_image_dir,
                prompt_csv=unpaired_prompt_csv,
                out_size=unpaired_out_size,
                crop_type=unpaired_crop_type,
                fading_params=fading_params if fading_params else None,
                random_fading=random_fading,
            )
            print(f"Unpaired dataset: {len(self.unpaired_dataset)} images")

        # 初始化paired数据集
        self.paired_dataset = None
        if paired_data_root is not None and paired_ratio > 0.0:
            self.paired_dataset = PairedFadingDataset(
                data_root=paired_data_root,
                prompt_csv=paired_prompt_csv,
                out_size=paired_out_size,
                crop_type=paired_crop_type,
                ori_subdir=paired_ori_subdir,
                fixed_subdir=paired_fixed_subdir,
            )
            print(f"Paired dataset: {len(self.paired_dataset)} image pairs")

        # 验证至少有一个数据集
        if self.unpaired_dataset is None and self.paired_dataset is None:
            raise ValueError("至少需要提供unpaired或paired数据集之一")

        # 计算总样本数
        if total_samples is not None:
            self._length = total_samples
        else:
            unpaired_len = len(self.unpaired_dataset) if self.unpaired_dataset else 0
            paired_len = len(self.paired_dataset) if self.paired_dataset else 0
            self._length = unpaired_len + paired_len

        # 打印混合信息
        print(f"\nMixedFadingDataset initialized:")
        print(f"  - Paired ratio: {self.paired_ratio:.1%}")
        print(f"  - Total samples per epoch: {self._length}")
        if self.unpaired_dataset and self.paired_dataset:
            expected_unpaired = int(self._length * (1 - self.paired_ratio))
            expected_paired = self._length - expected_unpaired
            print(f"  - Expected unpaired samples: ~{expected_unpaired}")
            print(f"  - Expected paired samples: ~{expected_paired}")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        根据paired_ratio随机选择从哪个数据集采样

        Returns:
            gt: ground truth图片 [H, W, 3], float32, range [-1, 1]
            lq: 低质量图片（褪色） [H, W, 3], float32, range [0, 1]
            prompt: 图片描述文本
        """
        # 只有unpaired数据集
        if self.paired_dataset is None:
            real_idx = idx % len(self.unpaired_dataset)
            return self.unpaired_dataset[real_idx]

        # 只有paired数据集
        if self.unpaired_dataset is None:
            real_idx = idx % len(self.paired_dataset)
            return self.paired_dataset[real_idx]

        # 混合采样
        if np.random.random() < self.paired_ratio:
            # 从paired数据集采样
            real_idx = np.random.randint(0, len(self.paired_dataset))
            return self.paired_dataset[real_idx]
        else:
            # 从unpaired数据集采样
            real_idx = np.random.randint(0, len(self.unpaired_dataset))
            return self.unpaired_dataset[real_idx]


class BalancedMixedFadingDataset(Dataset):
    """
    平衡混合褪色数据集

    与MixedFadingDataset的区别：
    - MixedFadingDataset: 每次采样随机选择数据集
    - BalancedMixedFadingDataset: 确保每个epoch中两种数据的比例精确

    适用于需要精确控制数据比例的场景
    """

    def __init__(
        self,
        # Unpaired数据集参数
        unpaired_image_dir: str = None,
        unpaired_prompt_csv: str = None,
        unpaired_out_size: int = 512,
        unpaired_crop_type: str = "resize",
        # 褪色参数（用于unpaired）- 新格式
        fading_params: dict = None,
        random_fading: bool = True,
        # 褪色参数（用于unpaired）- 旧格式（兼容）
        aging_type: str = None,
        decay_range: tuple = None,
        darken_strength: float = None,
        use_brown_overlay: bool = None,
        overlay_opacity: float = None,
        # Paired数据集参数
        paired_data_root: str = None,
        paired_prompt_csv: str = None,
        paired_out_size: int = 512,
        paired_crop_type: str = "resize",
        paired_ori_subdir: str = "ori",
        paired_fixed_subdir: str = "fixed",
        # 混合参数
        paired_ratio: float = 0.5,
        total_samples: int = 1000,  # 必须指定
    ):
        """
        Args:
            paired_ratio: paired数据的精确比例
            total_samples: 每个epoch的总样本数（必须指定）
        """
        self.paired_ratio = paired_ratio
        self._length = total_samples

        # 处理褪色参数：支持新旧两种格式
        if fading_params is None:
            fading_params = {}

        if aging_type is not None:
            fading_params['aging_type'] = aging_type
        if decay_range is not None:
            fading_params['decay_range'] = decay_range
        if darken_strength is not None:
            fading_params['darken_strength'] = darken_strength
        if use_brown_overlay is not None:
            fading_params['use_brown_overlay'] = use_brown_overlay
        if overlay_opacity is not None:
            fading_params['overlay_opacity'] = overlay_opacity

        # 计算每种数据的样本数
        self.num_paired = int(total_samples * paired_ratio)
        self.num_unpaired = total_samples - self.num_paired

        # 初始化unpaired数据集
        self.unpaired_dataset = None
        if unpaired_image_dir is not None and self.num_unpaired > 0:
            self.unpaired_dataset = FadingRestorationDataset(
                image_dir=unpaired_image_dir,
                prompt_csv=unpaired_prompt_csv,
                out_size=unpaired_out_size,
                crop_type=unpaired_crop_type,
                fading_params=fading_params if fading_params else None,
                random_fading=random_fading,
            )
            print(f"Unpaired dataset: {len(self.unpaired_dataset)} images")

        # 初始化paired数据集
        self.paired_dataset = None
        if paired_data_root is not None and self.num_paired > 0:
            self.paired_dataset = PairedFadingDataset(
                data_root=paired_data_root,
                prompt_csv=paired_prompt_csv,
                out_size=paired_out_size,
                crop_type=paired_crop_type,
                ori_subdir=paired_ori_subdir,
                fixed_subdir=paired_fixed_subdir,
            )
            print(f"Paired dataset: {len(self.paired_dataset)} image pairs")

        # 验证
        if self.num_unpaired > 0 and self.unpaired_dataset is None:
            raise ValueError("需要unpaired数据但未提供unpaired_image_dir")
        if self.num_paired > 0 and self.paired_dataset is None:
            raise ValueError("需要paired数据但未提供paired_data_root")

        # 生成索引映射
        self._generate_indices()

        print(f"\nBalancedMixedFadingDataset initialized:")
        print(f"  - Total samples: {self._length}")
        print(f"  - Unpaired samples: {self.num_unpaired} ({100*(1-paired_ratio):.1f}%)")
        print(f"  - Paired samples: {self.num_paired} ({100*paired_ratio:.1f}%)")

    def _generate_indices(self):
        """生成索引映射，确保比例精确"""
        indices = []

        # 添加unpaired索引
        if self.num_unpaired > 0:
            unpaired_indices = np.random.choice(
                len(self.unpaired_dataset),
                size=self.num_unpaired,
                replace=True
            )
            for idx in unpaired_indices:
                indices.append(('unpaired', idx))

        # 添加paired索引
        if self.num_paired > 0:
            paired_indices = np.random.choice(
                len(self.paired_dataset),
                size=self.num_paired,
                replace=True
            )
            for idx in paired_indices:
                indices.append(('paired', idx))

        # 打乱顺序
        np.random.shuffle(indices)
        self.indices = indices

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """根据预生成的索引获取样本"""
        dataset_type, real_idx = self.indices[idx]

        if dataset_type == 'paired':
            return self.paired_dataset[real_idx]
        else:
            return self.unpaired_dataset[real_idx]

    def reshuffle(self):
        """重新生成索引映射（可在每个epoch开始时调用）"""
        self._generate_indices()


if __name__ == "__main__":
    # 测试代码
    import os
    import csv
    from PIL import Image

    # 创建测试数据
    test_root = "/tmp/mixed_fading_test"

    # Unpaired数据
    unpaired_dir = os.path.join(test_root, "unpaired")
    os.makedirs(unpaired_dir, exist_ok=True)

    # Paired数据
    paired_dir = os.path.join(test_root, "paired")
    ori_dir = os.path.join(paired_dir, "ori")
    fixed_dir = os.path.join(paired_dir, "fixed")
    os.makedirs(ori_dir, exist_ok=True)
    os.makedirs(fixed_dir, exist_ok=True)

    # 创建测试图片
    for i in range(5):
        # Unpaired
        img = np.random.randint(100, 255, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(unpaired_dir, f"unpaired_{i}.jpg"))

        # Paired
        ori_img = np.random.randint(50, 150, (256, 256, 3), dtype=np.uint8)
        fixed_img = np.random.randint(100, 255, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(ori_img).save(os.path.join(ori_dir, f"paired_{i}.jpg"))
        Image.fromarray(fixed_img).save(os.path.join(fixed_dir, f"paired_{i}.jpg"))

    # 创建CSV
    unpaired_csv = os.path.join(test_root, "unpaired_prompts.csv")
    paired_csv = os.path.join(test_root, "paired_prompts.csv")

    with open(unpaired_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'prompt'])
        for i in range(5):
            writer.writerow([f'unpaired_{i}.jpg', f'Unpaired image {i}'])

    with open(paired_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'prompt'])
        for i in range(5):
            writer.writerow([f'paired_{i}.jpg', f'Paired image {i}'])

    # 测试MixedFadingDataset
    print("=" * 50)
    print("Testing MixedFadingDataset")
    print("=" * 50)

    dataset = MixedFadingDataset(
        unpaired_image_dir=unpaired_dir,
        unpaired_prompt_csv=unpaired_csv,
        unpaired_out_size=128,
        unpaired_crop_type="resize",
        paired_data_root=paired_dir,
        paired_prompt_csv=paired_csv,
        paired_out_size=128,
        paired_crop_type="resize",
        paired_ratio=0.5,
        total_samples=20,
    )

    print(f"\nDataset length: {len(dataset)}")

    # 统计采样比例
    paired_count = 0
    for i in range(len(dataset)):
        gt, lq, prompt = dataset[i]
        # 可以通过prompt来区分
        if 'Paired' in prompt:
            paired_count += 1

    print(f"Actual paired ratio: {paired_count}/{len(dataset)} = {paired_count/len(dataset):.1%}")

    # 测试BalancedMixedFadingDataset
    print("\n" + "=" * 50)
    print("Testing BalancedMixedFadingDataset")
    print("=" * 50)

    balanced_dataset = BalancedMixedFadingDataset(
        unpaired_image_dir=unpaired_dir,
        unpaired_prompt_csv=unpaired_csv,
        unpaired_out_size=128,
        unpaired_crop_type="resize",
        paired_data_root=paired_dir,
        paired_prompt_csv=paired_csv,
        paired_out_size=128,
        paired_crop_type="resize",
        paired_ratio=0.3,
        total_samples=20,
    )

    print(f"\nDataset length: {len(balanced_dataset)}")

    # 统计采样比例
    paired_count = 0
    for i in range(len(balanced_dataset)):
        gt, lq, prompt = balanced_dataset[i]
        if 'Paired' in prompt:
            paired_count += 1

    print(f"Actual paired ratio: {paired_count}/{len(balanced_dataset)} = {paired_count/len(balanced_dataset):.1%}")

    # 清理
    import shutil
    shutil.rmtree(test_root)
    print("\nTest data cleaned up")
