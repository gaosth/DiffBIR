"""
最小化测试：检查是否存在色彩偏移
直接测试RGB<->BGR转换和各个函数
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def test_rgb_bgr_conversion():
    """测试RGB<->BGR转换是否完全可逆"""
    print("=" * 80)
    print("测试1: RGB<->BGR转换")
    print("=" * 80)

    # 创建测试图像
    test_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # RGB -> BGR -> RGB
    bgr = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
    rgb_back = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    diff = np.abs(test_img.astype(np.int16) - rgb_back.astype(np.int16))
    max_diff = diff.max()

    print(f"原图与转换后的差异: max={max_diff}, mean={diff.mean():.6f}")
    print(f"是否完全相同: {np.array_equal(test_img, rgb_back)}")

    return max_diff == 0


def test_float_conversion():
    """测试uint8<->float32转换"""
    print("\n" + "=" * 80)
    print("测试2: uint8<->float32转换")
    print("=" * 80)

    test_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # uint8 -> float32 -> uint8
    float_img = test_img.astype(np.float32)
    uint8_back = float_img.astype(np.uint8)

    print(f"是否完全相同: {np.array_equal(test_img, uint8_back)}")

    # 测试带运算的情况
    float_img2 = test_img.astype(np.float32)
    float_img2 = float_img2 * 1.0  # 乘以1.0
    uint8_back2 = np.clip(float_img2, 0, 255).astype(np.uint8)

    diff = np.abs(test_img.astype(np.int16) - uint8_back2.astype(np.int16))
    print(f"经过×1.0运算后的差异: max={diff.max()}, mean={diff.mean():.6f}")

    return np.array_equal(test_img, uint8_back2)


def test_color_decay_with_1_0():
    """测试decay_range=(1.0, 1.0)时是否有变化"""
    print("\n" + "=" * 80)
    print("测试3: _apply_color_decay with decay_range=(1.0, 1.0)")
    print("=" * 80)

    # 导入dataset以使用其方法
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from diffbir.dataset.fading_restoration import FadingRestorationDataset

    # 创建一个dummy dataset对象
    class DummyDataset:
        def _apply_color_decay(self, image, decay_range):
            h, w = image.shape[:2]
            result = image.copy().astype(np.float32)

            # 创建衰减mask
            decay_mask = np.ones((h, w), dtype=np.float32)
            num_regions = 5

            for _ in range(num_regions):
                center_x = np.random.randint(0, w)
                center_y = np.random.randint(0, h)
                decay_strength = np.random.uniform(decay_range[0], decay_range[1])
                radius = np.random.randint(min(h, w) // 4, min(h, w) // 2)

                y_grid, x_grid = np.ogrid[:h, :w]
                distance = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
                regional_decay = np.exp(-(distance**2) / (2 * (radius**2)))
                regional_decay = 1 - regional_decay * (1 - decay_strength)
                decay_mask = np.minimum(decay_mask, regional_decay)

            # 平滑过渡
            decay_mask = cv2.GaussianBlur(decay_mask, (51, 51), 0)

            # 应用到每个通道
            for c in range(3):
                result[:, :, c] *= decay_mask

            return np.clip(result, 0, 255)

    dataset = DummyDataset()

    # 固定随机种子以便复现
    np.random.seed(42)

    test_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8).astype(np.float32)
    result = dataset._apply_color_decay(test_img, (1.0, 1.0))

    diff = np.abs(test_img - result)
    print(f"decay_range=(1.0, 1.0)的差异: max={diff.max():.6f}, mean={diff.mean():.6f}")
    print(f"理论上应该完全相同（因为decay_strength=1.0时不衰减）")

    # 检查mask的值
    print(f"\n检查mask生成过程:")
    print(f"  如果decay_strength=1.0, regional_decay = 1 - exp(...) * (1-1.0) = 1 - 0 = 1")
    print(f"  所以mask应该全是1.0")

    return diff.max() < 1e-5


def test_brown_overlay_with_0():
    """测试opacity=0时是否有变化"""
    print("\n" + "=" * 80)
    print("测试4: _apply_brown_overlay_only with opacity=0")
    print("=" * 80)

    test_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8).astype(np.float32)

    # 模拟brown overlay操作
    overlay_opacity = 0.0
    result = test_img.copy().astype(np.float32)
    brown_color = np.array([15, 49, 80], dtype=np.float32)
    brown_layer = np.ones_like(result) * brown_color
    result = result * (1 - overlay_opacity) + brown_layer * overlay_opacity
    result = np.clip(result, 0, 255)

    diff = np.abs(test_img - result)
    print(f"opacity=0的差异: max={diff.max():.6f}, mean={diff.mean():.6f}")

    return diff.max() < 1e-5


def test_full_pipeline_passthrough():
    """测试完整pipeline在禁用所有效果时是否有变化"""
    print("\n" + "=" * 80)
    print("测试5: 完整pipeline（所有效果禁用）")
    print("=" * 80)

    # 创建测试图像
    test_img_rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    # 模拟apply_fading_degradation的过程
    image_bgr = cv2.cvtColor(test_img_rgb, cv2.COLOR_RGB2BGR)

    # 模拟_synthesize_degradation（所有效果禁用）
    result = image_bgr.copy().astype(np.float32)

    # aging_type='darken', darken_strength=0.0 -> 跳过
    # decay_range=(1.0, 1.0) -> 应该不变
    # opacity=0 -> 应该不变
    # noise=0 -> 跳过

    result = np.clip(result, 0, 255).astype(np.uint8)

    # 转换回RGB
    degraded_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    diff = np.abs(test_img_rgb.astype(np.int16) - degraded_rgb.astype(np.int16))
    print(f"完整pipeline的差异: max={diff.max()}, mean={diff.mean():.6f}")
    print(f"是否完全相同: {np.array_equal(test_img_rgb, degraded_rgb)}")

    if diff.max() > 0:
        print(f"\n⚠️ 发现差异！")
        print(f"R通道差异: max={diff[:,:,0].max()}, mean={diff[:,:,0].mean():.6f}")
        print(f"G通道差异: max={diff[:,:,1].max()}, mean={diff[:,:,1].mean():.6f}")
        print(f"B通道差异: max={diff[:,:,2].max()}, mean={diff[:,:,2].mean():.6f}")

    return diff.max() == 0


if __name__ == "__main__":
    results = []

    results.append(("RGB<->BGR转换", test_rgb_bgr_conversion()))
    results.append(("uint8<->float32转换", test_float_conversion()))
    results.append(("color_decay(1.0,1.0)", test_color_decay_with_1_0()))
    results.append(("brown_overlay(0)", test_brown_overlay_with_0()))
    results.append(("完整pipeline", test_full_pipeline_passthrough()))

    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    if all(result[1] for result in results):
        print("\n✓ 所有测试通过！理论上不应该有色彩偏移。")
    else:
        print("\n✗ 部分测试失败，需要进一步调查。")
