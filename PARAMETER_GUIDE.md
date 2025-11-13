# 褪色参数详细指南

## 问题：为什么关闭所有效果后颜色还是变化很多？

### 答案：色彩衰减 (decay_range)

即使你设置了：
- `aging_type='darken'`（跳过褪色）
- `darken_strength=0`（不做HSV变暗）
- `use_brown_overlay=False`（不叠加棕色）

**图片还是会变暗**，原因是：

```python
# 这个步骤总是会执行！
result = self._apply_color_decay(result, params['decay_range'])
```

`decay_range` 参数会将图片的所有RGB通道**乘以一个系数**。

## decay_range 参数说明

### 格式
```python
decay_range: (min_value, max_value)
```

### 效果

| decay_range | 效果 | 亮度保留 |
|-------------|------|----------|
| `(0.5, 0.7)` | **强衰减** | 50%-70% |
| `(0.7, 0.85)` | 中强衰减 | 70%-85% |
| `(0.8, 0.95)` | **中度衰减（推荐）** | 80%-95% |
| `(0.9, 1.0)` | 轻微衰减 | 90%-100% |
| `(0.95, 1.0)` | 几乎无衰减 | 95%-100% |
| `(1.0, 1.0)` | **完全禁用** | 100% |

### 计算公式

```python
# 在 _apply_color_decay 中
decay_mask = [0.5-0.7之间的随机值]  # 根据decay_range
result = image * decay_mask
```

例如：
- `decay_range=(0.5, 0.7)`：每个像素会乘以 0.5-0.7
- 原始值 200 → 变为 100-140（明显变暗）
- `decay_range=(0.95, 1.0)`：每个像素会乘以 0.95-1.0
- 原始值 200 → 变为 190-200（几乎不变）

## 完整参数列表及其作用范围

### aging_type='fade' 时生效
```python
saturation: 0.6    # 饱和度降低到60%
brightness: 1.4    # 亮度提升40%
yellow: 0.6        # 添加60%黄色调
sepia: 0.4         # 添加40%棕褐色调
```

### aging_type='darken' 时生效
```python
darken_strength: 0.3  # HSV V通道降低30%
use_brown_overlay: true
overlay_opacity: 0.65 # 棕色叠加65%不透明度
```

### 总是生效（无论aging_type是什么）
```python
decay_range: [0.8, 0.95]  # 色彩衰减！
crack_density: 0.3
crack_thickness: 1
noise_level: 10
num_stains: 5
```

## 实用配置示例

### 配置1：纯净棕色叠加（推荐测试用）

```python
fading_params = {
    'aging_type': 'darken',
    'darken_strength': 0.0,       # 不做HSV变暗
    'use_brown_overlay': True,
    'overlay_opacity': 0.65,
    'decay_range': (1.0, 1.0),    # 完全禁用色彩衰减
    'crack_density': 0.0,
    'noise_level': 0,
    'num_stains': 0
}
```

**效果**：只有棕色叠加，其他完全不变

### 配置2：轻度老化（推荐训练用）

```python
fading_params = {
    'aging_type': 'darken',
    'darken_strength': 0.3,
    'use_brown_overlay': True,
    'overlay_opacity': 0.65,
    'decay_range': (0.8, 0.95),   # 中度衰减
    'crack_density': 0.3,
    'noise_level': 10,
    'num_stains': 5
}
```

**效果**：轻度老化，保留80%-95%亮度

### 配置3：重度老化

```python
fading_params = {
    'aging_type': 'both',          # 同时应用褪色和变暗
    'saturation': 0.5,
    'brightness': 1.5,
    'yellow': 0.7,
    'sepia': 0.5,
    'darken_strength': 0.4,
    'use_brown_overlay': True,
    'overlay_opacity': 0.75,
    'decay_range': (0.5, 0.7),     # 强衰减
    'crack_density': 0.4,
    'noise_level': 15,
    'num_stains': 8
}
```

**效果**：重度老化，只保留50%-70%亮度

## 测试脚本

### 测试纯净棕色叠加

```bash
python test_pure_brown_overlay.py \
    --image_dir /path/to/images \
    --prompt_csv /path/to/prompts.csv \
    --opacity 0.65
```

这个脚本会：
- 完全禁用所有其他效果
- 只应用棕色叠加
- decay_range 设为 (1.0, 1.0)

### 测试不同decay_range的效果

手动修改 `test_fading_dataset.py` 中的 `decay_range` 并运行：

```python
# 测试1: 无衰减
'decay_range': (1.0, 1.0)

# 测试2: 轻微衰减
'decay_range': (0.95, 1.0)

# 测试3: 中度衰减
'decay_range': (0.8, 0.95)

# 测试4: 强衰减
'decay_range': (0.5, 0.7)
```

## 训练建议

根据你的数据特点选择：

### 如果你的古画本身已经比较暗
```yaml
decay_range: [0.9, 1.0]   # 轻微衰减
```

### 如果你的古画亮度正常
```yaml
decay_range: [0.8, 0.95]  # 中度衰减（推荐）
```

### 如果你的古画很鲜艳
```yaml
decay_range: [0.6, 0.8]   # 较强衰减
```

## 调试技巧

1. **先禁用所有效果**，逐个开启：
   ```python
   decay_range: (1.0, 1.0)
   darken_strength: 0.0
   use_brown_overlay: False
   crack_density: 0.0
   noise_level: 0
   num_stains: 0
   ```

2. **使用诊断脚本**查看每步的效果：
   ```bash
   python diagnose_fading_pipeline.py \
       --image_dir /path/to/images \
       --prompt_csv /path/to/prompts.csv
   ```

3. **使用固定参数**（禁用随机化）便于对比：
   ```python
   random_fading: False
   ```

## 常见误区

❌ **误区1**：以为 `aging_type='darken'` 就不会变暗
- 事实：`decay_range` 总是会执行，会让图片变暗

❌ **误区2**：以为 `darken_strength=0` 就没有变暗效果
- 事实：`decay_range` 是独立的，会降低亮度

❌ **误区3**：以为只有棕色叠加会改变颜色
- 事实：`decay_range` 会降低所有通道的值

✅ **正确理解**：
```
总亮度变化 = HSV变暗 × 色彩衰减 × 棕色叠加

其中：
- HSV变暗：由 darken_strength 控制（aging_type='darken'时）
- 色彩衰减：由 decay_range 控制（总是生效）
- 棕色叠加：由 overlay_opacity 控制（aging_type='darken'时）
```

## 已修复的Bug

### Bug #1: HSV转换导致的颜色偏移（已修复）🔥

**症状**：即使 `darken_strength` 设置为很小的值（如0.1），图片仍然出现明显的红色偏移。

**根本原因**：
- `_apply_darkening()` 使用HSV色彩空间转换来实现变暗效果
- BGR ↔ HSV 转换涉及三角函数和非线性计算，会累积浮点误差
- HSV色彩空间的圆柱形结构导致转换不完全可逆
- 红色位于HSV色调环的0°/360°边界，对舍入误差特别敏感
- 即使只修改V通道（亮度），转换回BGR时也可能影响颜色

**旧的实现**（有颜色偏移）：
```python
# BGR → HSV → 修改V通道 → HSV → BGR
hsv = cv2.cvtColor(result / 255.0, cv2.COLOR_BGR2HSV)
hsv[:, :, 2] *= (1 - darken_strength)  # 降低亮度
hsv[:, :, 1] *= 1.4  # 提升饱和度（加剧颜色偏移）
result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) * 255
```

**新的实现**（无颜色偏移）：
```python
# 直接线性变暗，所有通道同比例降低
result = image * (1 - darken_strength)
```

**技术优势**：
- ✅ 无色彩空间转换 → 无转换误差
- ✅ 线性操作 → 可预测，可逆
- ✅ 所有通道同比例 → 保持颜色不变
- ✅ 性能更好 → 避免昂贵的三角函数运算

**影响**：
- 修复后，任何 `darken_strength` 值都不会产生颜色偏移
- 只降低亮度，完全保持原始颜色的色调和饱和度
- `test_pure_brown_overlay.py` 和所有测试脚本现在都能正确工作

### Bug #2: Early return位置错误（已修复）

**症状**：即使设置了 `darken_strength=0.0`，仍然有轻微的类型转换误差。

**根本原因**：
- Early return 检查在类型转换**之后**
- 即使跳过处理，仍然执行了 `uint8 → float32 → clip → uint8`

**修复方案**：
```python
def _apply_darkening(self, ...):
    # 先检查，完全跳过所有操作
    if darken_strength == 0.0:
        return image  # 直接返回，无任何转换

    result = image.astype(np.float32)  # 只有需要时才转换
    ...
```

**影响**：
- `darken_strength=0.0` 时零开销，完全绕过函数
- 同样应用于 `_apply_color_decay()` 和 `_apply_brown_overlay_only()`

## 快速参考

想要什么效果？

| 需求 | decay_range | darken_strength | overlay_opacity |
|------|-------------|-----------------|-----------------|
| 只要棕色叠加 | (1.0, 1.0) | 0.0 | 0.5-0.8 |
| 轻微老化 | (0.9, 1.0) | 0.2 | 0.5-0.7 |
| 中度老化 | (0.8, 0.95) | 0.3 | 0.6-0.75 |
| 重度老化 | (0.6, 0.8) | 0.4 | 0.7-0.85 |

希望这个指南能帮你理解各个参数的作用！
