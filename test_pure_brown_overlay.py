"""
纯净的棕色叠加测试
不应用任何其他效果，只测试棕色叠加本身
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from diffbir.dataset.fading_restoration import FadingRestorationDataset


def test_pure_brown_overlay(image_dir, prompt_csv, overlay_opacity=0.65):
    """
    测试纯净的棕色叠加效果（不应用任何其他效果）

    Args:
        image_dir: 图片目录
        prompt_csv: CSV文件路径
        overlay_opacity: 棕色叠加不透明度
    """
    print("=" * 80)
    print("纯净棕色叠加测试")
    print("=" * 80)

    # 最小化的参数：只保留棕色叠加
    fading_params = {
        'saturation': 1.0,        # 不改变
        'brightness': 1.0,        # 不改变
        'yellow': 0.0,            # 不添加
        'sepia': 0.0,             # 不添加
        'crack_density': 0.0,     # 不添加
        'crack_thickness': 1,
        'crack_type': 'light',
        'decay_range': (1.0, 1.0),  # 不衰减！
        'noise_level': 0,           # 不添加噪声
        'num_stains': 0,            # 不添加污渍
        'aging_type': 'darken',
        'darken_strength': 0.0,     # 不做HSV变暗！
        'use_brown_overlay': True,
        'overlay_opacity': overlay_opacity
    }

    print(f"\n配置:")
    print(f"  棕色叠加不透明度: {overlay_opacity}")
    print(f"  其他所有效果: 已禁用")

    # 创建数据集
    dataset = FadingRestorationDataset(
        image_dir=image_dir,
        prompt_csv=prompt_csv,
        out_size=512,
        crop_type="center",
        random_fading=False,
        fading_params=fading_params
    )

    print(f"\n✓ 数据集创建成功")
    print(f"  数据集大小: {len(dataset)}")

    # 加载样本
    gt, lq, prompt = dataset[0]

    # 转换显示格式
    gt_vis = (gt + 1) / 2
    lq_vis = lq

    # 创建对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    axes[0].imshow(gt_vis)
    axes[0].set_title(f"原始图片", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(lq_vis)
    axes[1].set_title(f"仅棕色叠加 (opacity={overlay_opacity})", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.suptitle(f"Prompt: {prompt}\n\n"
                f"纯净棕色叠加测试 - 无其他效果",
                fontsize=11)
    plt.tight_layout()

    output_path = f"pure_brown_overlay_opacity_{overlay_opacity:.2f}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 对比图保存到: {output_path}")
    plt.close()

    # 颜色分析
    print("\n" + "=" * 80)
    print("颜色分析")
    print("=" * 80)

    def analyze(img, name):
        r, g, b = img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()
        brown_index = b - r
        print(f"\n{name}:")
        print(f"  RGB均值: R={r:.3f}, G={g:.3f}, B={b:.3f}")
        print(f"  棕色指数 (B-R): {brown_index:.3f}")

    analyze(gt_vis, "原始图片")
    analyze(lq_vis, "棕色叠加后")

    print("\n棕色叠加色值: #50310f (RGB: [80, 49, 15])")
    print(f"混合公式: 结果 = 原图 × (1 - {overlay_opacity}) + 棕色 × {overlay_opacity}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="测试纯净的棕色叠加效果")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--prompt_csv", type=str, required=True)
    parser.add_argument("--opacity", type=float, default=0.65,
                       help="棕色叠加不透明度 (0.0-1.0)")

    args = parser.parse_args()

    if not os.path.exists(args.image_dir):
        print(f"错误: 图片目录不存在: {args.image_dir}")
        sys.exit(1)

    if not os.path.exists(args.prompt_csv):
        print(f"错误: CSV文件不存在: {args.prompt_csv}")
        sys.exit(1)

    test_pure_brown_overlay(args.image_dir, args.prompt_csv, args.opacity)
