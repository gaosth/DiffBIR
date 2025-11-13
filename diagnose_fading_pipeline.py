"""
诊断脚本：逐步显示褪色效果的各个阶段
帮助理解为什么棕色叠加效果不明显
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from diffbir.dataset.fading_restoration import FadingRestorationDataset


def diagnose_fading_pipeline(image_dir, prompt_csv):
    """
    逐步展示褪色pipeline的每个阶段
    """
    print("=" * 80)
    print("褪色效果Pipeline诊断")
    print("=" * 80)

    # 创建数据集
    fading_params = {
        'saturation': 0.6,
        'brightness': 1.4,
        'yellow': 0.6,
        'sepia': 0.4,
        'crack_density': 0.0,
        'crack_thickness': 1,
        'crack_type': 'light',
        'decay_range': (0.5, 0.7),
        'noise_level': 0,  # 禁用噪声
        'num_stains': 0,    # 禁用污渍
        'aging_type': 'both',
        'darken_strength': 0.3,
        'use_brown_overlay': True,
        'overlay_opacity': 0.65
    }

    dataset = FadingRestorationDataset(
        image_dir=image_dir,
        prompt_csv=prompt_csv,
        out_size=512,
        crop_type="center",
        random_fading=False,
        fading_params=fading_params
    )

    # 手动执行每个步骤
    print("\n手动执行褪色pipeline的每个步骤...")

    # 加载原图
    filename = dataset.image_files[0]
    img_gt = dataset.load_and_crop_image(filename)
    img_bgr = cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR).astype(np.float32)

    steps = []
    steps.append(("0. 原始图片", img_bgr.copy()))

    # 步骤1: 褪色
    result = img_bgr.copy()
    if fading_params['aging_type'] in ['fade', 'both']:
        result = dataset._apply_fading(
            result,
            saturation=fading_params['saturation'],
            brightness=fading_params['brightness'],
            yellow=fading_params['yellow'],
            sepia=fading_params['sepia']
        )
    steps.append(("1. 应用褪色 (HSV调整)", result.copy()))

    # 步骤2: 变暗老化 (包括棕色叠加)
    if fading_params['aging_type'] in ['darken', 'both']:
        result = dataset._apply_darkening(
            result,
            darken_strength=fading_params['darken_strength'],
            use_overlay=fading_params['use_brown_overlay'],
            overlay_opacity=fading_params['overlay_opacity']
        )
    steps.append(("2. 应用棕色叠加 (#50310f)", result.copy()))

    # 步骤3: 色彩衰减 (这一步会削弱棕色效果！)
    result = dataset._apply_color_decay(result, fading_params['decay_range'])
    steps.append(("3. 应用色彩衰减 (削弱效果！)", result.copy()))

    # 创建对比图
    n_steps = len(steps)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (title, image) in enumerate(steps):
        # BGR转RGB用于显示
        img_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].axis('off')

        # 显示颜色统计
        img_norm = img_rgb.astype(np.float32) / 255.0
        r_mean = img_norm[:,:,0].mean()
        g_mean = img_norm[:,:,1].mean()
        b_mean = img_norm[:,:,2].mean()
        brown_index = b_mean - r_mean  # 棕色指数

        info_text = f"R:{r_mean:.3f} G:{g_mean:.3f} B:{b_mean:.3f}\n"
        info_text += f"棕色指数(B-R): {brown_index:.3f}"
        axes[idx].text(0.02, 0.98, info_text,
                      transform=axes[idx].transAxes,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                      fontsize=9)

    # 隐藏多余的subplot
    for idx in range(n_steps, len(axes)):
        axes[idx].axis('off')

    plt.suptitle("褪色效果Pipeline各阶段对比\n"
                "关键问题：步骤3的色彩衰减会削弱步骤2的棕色叠加效果",
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = "diagnosis_fading_pipeline.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ 诊断图保存到: {output_path}")
    plt.close()

    # 详细分析
    print("\n" + "=" * 80)
    print("颜色变化分析")
    print("=" * 80)

    for idx, (title, image) in enumerate(steps):
        img_rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0

        print(f"\n{title}:")
        print(f"  RGB均值: R={img_norm[:,:,0].mean():.3f}, "
              f"G={img_norm[:,:,1].mean():.3f}, "
              f"B={img_norm[:,:,2].mean():.3f}")

        brown_index = img_norm[:,:,2].mean() - img_norm[:,:,0].mean()
        print(f"  棕色指数 (B-R): {brown_index:.3f}")

        if idx == 2:
            print("  ↑ 棕色叠加后，棕色指数应该明显提高")
        elif idx == 3:
            print("  ↑ 色彩衰减后，棕色指数被削弱")

    print("\n" + "=" * 80)
    print("问题诊断")
    print("=" * 80)
    print("\n原因：")
    print("  步骤2添加了棕色叠加 (#50310f，RGB约为[80,49,15])")
    print("  步骤3的色彩衰减会对所有通道乘以0.5-0.7的系数")
    print("  这会削弱刚刚添加的棕色调！")

    print("\n建议的解决方案：")
    print("  1. 调整顺序：在色彩衰减之后再应用棕色叠加")
    print("  2. 或者：在测试时禁用色彩衰减")
    print("  3. 或者：增加棕色叠加的不透明度 (overlay_opacity > 0.8)")
    print("  4. 或者：减少色彩衰减的强度 (decay_range = (0.7, 0.9))")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="诊断褪色效果pipeline")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--prompt_csv", type=str, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.image_dir):
        print(f"错误: 图片目录不存在: {args.image_dir}")
        sys.exit(1)

    if not os.path.exists(args.prompt_csv):
        print(f"错误: CSV文件不存在: {args.prompt_csv}")
        sys.exit(1)

    diagnose_fading_pipeline(args.image_dir, args.prompt_csv)
