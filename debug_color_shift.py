"""
调试色彩偏移的根本原因
"""

import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from diffbir.dataset.fading_restoration import FadingRestorationDataset


def test_minimal_pipeline(image_dir, prompt_csv):
    """
    测试最小化pipeline，找出色彩偏移的根源
    """
    print("=" * 80)
    print("调试色彩偏移")
    print("=" * 80)

    # 创建dataset
    fading_params = {
        'saturation': 1.0,
        'brightness': 1.0,
        'yellow': 0.0,
        'sepia': 0.0,
        'crack_density': 0.0,
        'crack_thickness': 1,
        'crack_type': 'light',
        'decay_range': (1.0, 1.0),
        'noise_level': 0,
        'num_stains': 0,
        'aging_type': 'darken',
        'darken_strength': 0.0,
        'use_brown_overlay': True,
        'overlay_opacity': 0.0  # 完全禁用
    }

    dataset = FadingRestorationDataset(
        image_dir=image_dir,
        prompt_csv=prompt_csv,
        out_size=512,
        crop_type="center",
        random_fading=False,
        fading_params=fading_params
    )

    # 加载第一张图片
    filename = dataset.image_files[0]
    img_gt = dataset.load_and_crop_image(filename)  # RGB, uint8, [0,255]

    print(f"\n原始图片: {filename}")
    print(f"  Shape: {img_gt.shape}")
    print(f"  Dtype: {img_gt.dtype}")
    print(f"  Range: [{img_gt.min()}, {img_gt.max()}]")
    print(f"  RGB均值: R={img_gt[:,:,0].mean():.3f}, G={img_gt[:,:,1].mean():.3f}, B={img_gt[:,:,2].mean():.3f}")

    # 测试1: 直接通过apply_fading_degradation
    print("\n" + "-" * 80)
    print("测试1: 完整pipeline (所有效果禁用)")
    print("-" * 80)
    img_lq = dataset.apply_fading_degradation(img_gt)  # 应该返回RGB, uint8, [0,255]

    print(f"  Shape: {img_lq.shape}")
    print(f"  Dtype: {img_lq.dtype}")
    print(f"  Range: [{img_lq.min()}, {img_lq.max()}]")
    print(f"  RGB均值: R={img_lq[:,:,0].mean():.3f}, G={img_lq[:,:,1].mean():.3f}, B={img_lq[:,:,2].mean():.3f}")

    diff = np.abs(img_gt.astype(np.int16) - img_lq.astype(np.int16))
    print(f"\n  与原图差异:")
    print(f"    总体: max={diff.max()}, mean={diff.mean():.6f}")
    print(f"    R通道: max={diff[:,:,0].max()}, mean={diff[:,:,0].mean():.6f}")
    print(f"    G通道: max={diff[:,:,1].max()}, mean={diff[:,:,1].mean():.6f}")
    print(f"    B通道: max={diff[:,:,2].max()}, mean={diff[:,:,2].mean():.6f}")

    if diff.max() > 0:
        print(f"\n  ⚠️ 发现色彩偏移！")
        # 检查哪个通道偏移最大
        r_diff = diff[:,:,0].mean()
        g_diff = diff[:,:,1].mean()
        b_diff = diff[:,:,2].mean()

        if r_diff > max(g_diff, b_diff):
            print(f"  → R通道偏移最大 (偏红)")
        elif g_diff > max(r_diff, b_diff):
            print(f"  → G通道偏移最大 (偏绿)")
        else:
            print(f"  → B通道偏移最大 (偏蓝)")
    else:
        print(f"\n  ✓ 完全一致，无色彩偏移")

    # 测试2: 只测试RGB<->BGR转换
    print("\n" + "-" * 80)
    print("测试2: 仅RGB<->BGR转换")
    print("-" * 80)
    bgr = cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR)
    rgb_back = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    diff2 = np.abs(img_gt.astype(np.int16) - rgb_back.astype(np.int16))
    print(f"  差异: max={diff2.max()}, mean={diff2.mean():.6f}")
    print(f"  是否完全相同: {np.array_equal(img_gt, rgb_back)}")

    # 测试3: 只测试uint8<->float32转换
    print("\n" + "-" * 80)
    print("测试3: 仅uint8<->float32转换")
    print("-" * 80)
    float_img = img_gt.copy().astype(np.float32)
    uint8_back = np.clip(float_img, 0, 255).astype(np.uint8)

    diff3 = np.abs(img_gt.astype(np.int16) - uint8_back.astype(np.int16))
    print(f"  差异: max={diff3.max()}, mean={diff3.mean():.6f}")
    print(f"  是否完全相同: {np.array_equal(img_gt, uint8_back)}")

    # 测试4: 组合测试 (RGB->BGR->float32->clip->uint8->RGB)
    print("\n" + "-" * 80)
    print("测试4: 完整转换链 (模拟_synthesize_degradation)")
    print("-" * 80)

    # 模拟apply_fading_degradation
    image_bgr = cv2.cvtColor(img_gt, cv2.COLOR_RGB2BGR)

    # 模拟_synthesize_degradation (什么都不做)
    result = image_bgr.copy().astype(np.float32)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 转回RGB
    degraded_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    diff4 = np.abs(img_gt.astype(np.int16) - degraded_rgb.astype(np.int16))
    print(f"  差异: max={diff4.max()}, mean={diff4.mean():.6f}")
    print(f"  是否完全相同: {np.array_equal(img_gt, degraded_rgb)}")

    if diff4.max() > 0:
        print(f"\n  ⚠️ 组合转换链引入了误差！")
    else:
        print(f"\n  ✓ 组合转换链无误差")

    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)

    if diff.max() == 0:
        print("✓ 完整pipeline无色彩偏移 - 问题已修复！")
    else:
        print("✗ 仍存在色彩偏移")
        print(f"\n最可能的原因:")
        if diff4.max() > 0:
            print("  → RGB<->BGR<->float32<->uint8 转换链有问题")
        else:
            print("  → 某个效果函数未被正确跳过")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="调试色彩偏移")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--prompt_csv", type=str, required=True)

    args = parser.parse_args()

    test_minimal_pipeline(args.image_dir, args.prompt_csv)
