"""
测试褪色修复数据集
用于验证数据加载和预处理流程是否正常
"""

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from diffbir.dataset.fading_restoration import FadingRestorationDataset


def test_dataset(image_dir, prompt_csv, num_samples=5):
    """
    测试数据集加载

    Args:
        image_dir: 图片目录
        prompt_csv: CSV文件路径
        num_samples: 要测试的样本数量
    """
    print("=" * 80)
    print("测试褪色修复数据集")
    print("=" * 80)

    # 创建数据集
    try:
        dataset = FadingRestorationDataset(
            image_dir=image_dir,
            prompt_csv=prompt_csv,
            out_size=512,
            crop_type="center",
            random_fading=True
        )
        print(f"\n✓ 数据集创建成功")
        print(f"  数据集大小: {len(dataset)}")
    except Exception as e:
        print(f"\n✗ 数据集创建失败: {e}")
        return False

    # 测试加载样本
    print(f"\n测试加载 {num_samples} 个样本...")
    for i in range(min(num_samples, len(dataset))):
        try:
            gt, lq, prompt = dataset[i]
            print(f"\n样本 {i+1}:")
            print(f"  GT shape: {gt.shape}, range: [{gt.min():.3f}, {gt.max():.3f}]")
            print(f"  LQ shape: {lq.shape}, range: [{lq.min():.3f}, {lq.max():.3f}]")
            print(f"  Prompt: {prompt}")

            # 验证数据格式
            assert gt.shape == lq.shape, "GT和LQ的形状不匹配"
            assert gt.shape[0] == 512 and gt.shape[1] == 512, "图片尺寸不是512x512"
            assert gt.shape[2] == 3, "图片不是3通道"
            assert gt.min() >= -1 and gt.max() <= 1, "GT范围不在[-1, 1]"
            assert lq.min() >= 0 and lq.max() <= 1, "LQ范围不在[0, 1]"

        except Exception as e:
            print(f"\n✗ 加载样本 {i} 失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    print(f"\n✓ 所有样本加载成功")

    # 可视化第一个样本
    print(f"\n生成可视化图像...")
    try:
        gt, lq, prompt = dataset[0]

        # 转换数据格式用于显示
        # GT: [-1, 1] -> [0, 1]
        gt_vis = (gt + 1) / 2
        # LQ: [0, 1] -> [0, 1]
        lq_vis = lq

        # 创建对比图
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(lq_vis)
        axes[0].set_title("褪色后 (LQ - Input)", fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(gt_vis)
        axes[1].set_title("原始图片 (GT - Target)", fontsize=12)
        axes[1].axis('off')

        plt.suptitle(f"Prompt: {prompt}", fontsize=10, y=0.98)
        plt.tight_layout()

        # 保存图片
        output_path = "test_fading_sample.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ 可视化结果保存到: {output_path}")
        plt.close()

    except Exception as e:
        print(f"✗ 可视化失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="测试褪色修复数据集")
    parser.add_argument("--image_dir", type=str, required=True, help="图片目录路径")
    parser.add_argument("--prompt_csv", type=str, required=True, help="CSV文件路径")
    parser.add_argument("--num_samples", type=int, default=5, help="测试样本数量")

    args = parser.parse_args()

    # 检查路径是否存在
    if not os.path.exists(args.image_dir):
        print(f"错误: 图片目录不存在: {args.image_dir}")
        sys.exit(1)

    if not os.path.exists(args.prompt_csv):
        print(f"错误: CSV文件不存在: {args.prompt_csv}")
        sys.exit(1)

    # 运行测试
    success = test_dataset(args.image_dir, args.prompt_csv, args.num_samples)

    sys.exit(0 if success else 1)
