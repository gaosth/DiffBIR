"""
对比测试：棕色叠加效果 vs 无棕色叠加
生成并排对比图，展示use_brown_overlay参数的效果
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from diffbir.dataset.fading_restoration import FadingRestorationDataset


def compare_brown_overlay_effect(image_dir, prompt_csv):
    """
    对比棕色叠加效果

    Args:
        image_dir: 图片目录
        prompt_csv: CSV文件路径
    """
    print("=" * 80)
    print("棕色叠加效果对比测试")
    print("=" * 80)

    # 固定的褪色参数（用于公平对比）
    base_params = {
        'saturation': 0.6,
        'brightness': 1.4,
        'yellow': 0.6,
        'sepia': 0.4,
        'crack_density': 0.0,  # 禁用裂纹以便更清楚地看到颜色变化
        'crack_thickness': 1,
        'crack_type': 'light',
        'decay_range': (0.5, 0.7),
        'noise_level': 5,      # 降低噪声以便更清楚地看到颜色
        'num_stains': 0,       # 禁用污渍
        'aging_type': 'both',
        'darken_strength': 0.3,
        'overlay_opacity': 0.65
    }

    # 创建两个数据集：一个用棕色叠加，一个不用
    print("\n创建数据集...")

    # 数据集1: 不使用棕色叠加
    params_no_overlay = base_params.copy()
    params_no_overlay['use_brown_overlay'] = False

    dataset_no_overlay = FadingRestorationDataset(
        image_dir=image_dir,
        prompt_csv=prompt_csv,
        out_size=512,
        crop_type="center",
        random_fading=False,  # 使用固定参数以便对比
        fading_params=params_no_overlay
    )
    print(f"✓ 创建无棕色叠加数据集")

    # 数据集2: 使用棕色叠加
    params_with_overlay = base_params.copy()
    params_with_overlay['use_brown_overlay'] = True

    dataset_with_overlay = FadingRestorationDataset(
        image_dir=image_dir,
        prompt_csv=prompt_csv,
        out_size=512,
        crop_type="center",
        random_fading=False,
        fading_params=params_with_overlay
    )
    print(f"✓ 创建有棕色叠加数据集")

    # 加载第一个样本
    print("\n加载样本进行对比...")
    gt1, lq_no_overlay, prompt = dataset_no_overlay[0]
    gt2, lq_with_overlay, _ = dataset_with_overlay[0]

    # 转换显示格式
    gt_vis = (gt1 + 1) / 2
    lq_no_overlay_vis = lq_no_overlay
    lq_with_overlay_vis = lq_with_overlay

    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(gt_vis)
    axes[0].set_title("原始图片 (GT)", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(lq_no_overlay_vis)
    axes[1].set_title("无棕色叠加\n(HSV变暗方法)", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(lq_with_overlay_vis)
    axes[2].set_title("有棕色叠加 (#50310f)\n(HSV + 棕色层)", fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle(f"Prompt: {prompt}\n"
                f"Overlay Opacity: {base_params['overlay_opacity']}, "
                f"Darken Strength: {base_params['darken_strength']}",
                fontsize=11, y=0.98)
    plt.tight_layout()

    # 保存对比图
    output_path = "comparison_brown_overlay.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ 对比图保存到: {output_path}")
    plt.close()

    # 计算颜色统计信息
    print("\n" + "=" * 80)
    print("颜色统计分析")
    print("=" * 80)

    def analyze_color(image, name):
        # 转换为0-255范围用于统计
        img_uint8 = (image * 255).astype(np.uint8)
        print(f"\n{name}:")
        print(f"  RGB均值: R={img_uint8[:,:,0].mean():.1f}, "
              f"G={img_uint8[:,:,1].mean():.1f}, "
              f"B={img_uint8[:,:,2].mean():.1f}")
        print(f"  RGB标准差: R={img_uint8[:,:,0].std():.1f}, "
              f"G={img_uint8[:,:,1].std():.1f}, "
              f"B={img_uint8[:,:,2].std():.1f}")

        # 计算棕色调强度（棕色 #50310f 的BGR值为 [15, 49, 80]）
        # 高B值、中等G值、低R值表示棕色调
        brown_score = img_uint8[:,:,2].mean() - img_uint8[:,:,0].mean()
        print(f"  棕色调指数 (B-R): {brown_score:.1f} (值越大越偏棕色)")

    analyze_color(gt_vis, "原始图片")
    analyze_color(lq_no_overlay_vis, "无棕色叠加")
    analyze_color(lq_with_overlay_vis, "有棕色叠加")

    print("\n" + "=" * 80)
    print("对比完成！")
    print("=" * 80)
    print("\n关键差异说明:")
    print("  1. 无棕色叠加: 使用HSV方法降低亮度和调整饱和度")
    print("  2. 有棕色叠加: 在HSV基础上，额外叠加一层棕色 (#50310f)")
    print("     - 棕色叠加会使图片整体偏向暖棕色调")
    print("     - 更接近古画的泛黄老化效果")
    print("     - 不透明度可通过overlay_opacity参数调整 (当前: {:.2f})".format(
        base_params['overlay_opacity']))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="对比棕色叠加效果")
    parser.add_argument("--image_dir", type=str, required=True, help="图片目录路径")
    parser.add_argument("--prompt_csv", type=str, required=True, help="CSV文件路径")

    args = parser.parse_args()

    # 检查路径
    if not os.path.exists(args.image_dir):
        print(f"错误: 图片目录不存在: {args.image_dir}")
        sys.exit(1)

    if not os.path.exists(args.prompt_csv):
        print(f"错误: CSV文件不存在: {args.prompt_csv}")
        sys.exit(1)

    # 运行对比
    compare_brown_overlay_effect(args.image_dir, args.prompt_csv)
