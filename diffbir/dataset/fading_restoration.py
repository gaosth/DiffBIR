"""
褪色图片修复数据集
用于训练Stage2 ControlNet来修复褪色的古画
"""

from typing import Dict, Union, Tuple
import os
import csv
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data


class FadingRestorationDataset(data.Dataset):
    """
    褪色图片修复数据集

    数据组织方式：
    - 原始高质量图片存放在image_dir中
    - image_prompts.csv文件包含文件名和对应的prompt
    - 在线应用褪色效果生成低质量图片

    Args:
        image_dir: 原始图片目录路径
        prompt_csv: CSV文件路径，包含filename和prompt两列
        out_size: 输出图片尺寸
        crop_type: 裁剪类型 ['none', 'center', 'random']
        fading_params: 褪色参数字典
        random_fading: 是否随机化褪色参数以增加多样性
    """

    def __init__(
        self,
        image_dir: str,
        prompt_csv: str,
        out_size: int = 512,
        crop_type: str = "center",
        fading_params: dict = None,
        random_fading: bool = True,
    ):
        super(FadingRestorationDataset, self).__init__()
        self.image_dir = Path(image_dir)
        self.out_size = out_size
        self.crop_type = crop_type
        self.random_fading = random_fading

        assert crop_type in ["none", "center", "random"], \
            f"crop_type must be 'none', 'center' or 'random', got {crop_type}"

        # 默认褪色参数
        if fading_params is None:
            self.fading_params = {
                'saturation': 0.6,
                'brightness': 1.4,
                'yellow': 0.6,
                'sepia': 0.4,
                'crack_density': 0.3,
                'crack_thickness': 1,
                'crack_type': 'light',
                'decay_range': (0.5, 0.7),
                'noise_level': 10,
                'num_stains': 5,
                'aging_type': 'both',
                'darken_strength': 0.3,
                'use_brown_overlay': False,
                'overlay_opacity': 0.65
            }
        else:
            self.fading_params = fading_params

        # 读取prompt CSV文件
        self.prompts = self._load_prompts(prompt_csv)

        # 获取所有可用的图片文件
        self.image_files = self._load_image_files()

        print(f"FadingRestorationDataset initialized:")
        print(f"  Image directory: {self.image_dir}")
        print(f"  Total images: {len(self.image_files)}")
        print(f"  Output size: {self.out_size}")
        print(f"  Random fading: {self.random_fading}")

    def _load_prompts(self, prompt_csv: str) -> Dict[str, str]:
        """从CSV文件加载prompts"""
        prompts = {}
        with open(prompt_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row['filename']
                prompt = row['prompt']
                prompts[filename] = prompt
        print(f"Loaded {len(prompts)} prompts from {prompt_csv}")
        return prompts

    def _load_image_files(self):
        """加载所有在prompts中有记录的图片文件"""
        image_files = []
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        for filename in self.prompts.keys():
            image_path = self.image_dir / filename
            if image_path.exists() and image_path.suffix.lower() in supported_extensions:
                image_files.append(filename)
            else:
                print(f"Warning: Image {filename} not found or unsupported format")

        return image_files

    def load_and_crop_image(self, filename: str) -> np.ndarray:
        """加载并裁剪图片"""
        image_path = self.image_dir / filename
        image = Image.open(image_path).convert("RGB")

        if self.crop_type != "none":
            if image.height == self.out_size and image.width == self.out_size:
                image = np.array(image)
            else:
                if self.crop_type == "center":
                    image = self._center_crop(image, self.out_size)
                elif self.crop_type == "random":
                    image = self._random_crop(image, self.out_size)
        else:
            assert image.height == self.out_size and image.width == self.out_size, \
                f"Image size mismatch: expected {self.out_size}x{self.out_size}, got {image.height}x{image.width}"
            image = np.array(image)

        # hwc, rgb, 0-255, uint8
        return image

    def _center_crop(self, image: Image.Image, size: int) -> np.ndarray:
        """中心裁剪"""
        width, height = image.size
        left = (width - size) // 2
        top = (height - size) // 2
        right = left + size
        bottom = top + size

        if left < 0 or top < 0:
            # 如果图片太小，先resize
            scale = max(size / width, size / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            return self._center_crop(image, size)

        image = image.crop((left, top, right, bottom))
        return np.array(image)

    def _random_crop(self, image: Image.Image, size: int, min_crop_frac: float = 0.7) -> np.ndarray:
        """随机裁剪"""
        width, height = image.size
        min_size = int(size * min_crop_frac)

        if width < size or height < size:
            # 如果图片太小，先resize
            scale = max(size / width, size / height)
            new_width = int(width * scale * 1.1)  # 稍微放大一点以便裁剪
            new_height = int(height * scale * 1.1)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            width, height = image.size

        # 随机选择裁剪位置
        left = np.random.randint(0, max(1, width - size))
        top = np.random.randint(0, max(1, height - size))
        right = left + size
        bottom = top + size

        image = image.crop((left, top, right, bottom))
        return np.array(image)

    def apply_fading_degradation(self, image: np.ndarray) -> np.ndarray:
        """
        应用褪色退化效果

        Args:
            image: RGB图像，范围[0, 255]，uint8

        Returns:
            褪色后的图像，范围[0, 255]，uint8
        """
        # 转换为BGR格式（OpenCV格式）
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 获取褪色参数（可能随机化）
        if self.random_fading:
            params = self._get_random_fading_params()
        else:
            params = self.fading_params.copy()

        # 应用褪色效果（使用简化版本，只包含核心效果）
        degraded = self._synthesize_degradation(image_bgr, params)

        # 转换回RGB
        degraded_rgb = cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB)

        return degraded_rgb

    def _get_random_fading_params(self) -> dict:
        """
        生成随机化的褪色参数

        注意：以下参数会从fading_params中读取，不会随机化：
        - aging_type
        - use_brown_overlay
        - overlay_opacity
        - decay_range (新增：尊重用户设置)
        - crack_thickness
        - crack_type
        """
        # 如果用户设置了接近1.0的decay_range，说明不想要衰减，保持用户设置
        user_decay_min, user_decay_max = self.fading_params['decay_range']
        if user_decay_min >= 0.85:  # 用户希望保持高亮度
            decay_range = self.fading_params['decay_range']
            # 同时使用较小的 darken_strength 以保持亮度
            darken_strength = self.fading_params['darken_strength']
        else:
            # 否则使用随机范围
            decay_range = (np.random.uniform(0.4, 0.6), np.random.uniform(0.6, 0.8))
            darken_strength = np.random.uniform(0.2, 0.4)

        return {
            'saturation': np.random.uniform(0.5, 0.7),
            'brightness': np.random.uniform(1.2, 1.6),
            'yellow': np.random.uniform(0.4, 0.8),
            'sepia': np.random.uniform(0.3, 0.5),
            'crack_density': np.random.uniform(0.2, 0.4),
            'crack_thickness': self.fading_params['crack_thickness'],
            'crack_type': self.fading_params['crack_type'],
            'decay_range': decay_range,
            'noise_level': int(np.random.uniform(5, 15)),
            'num_stains': int(np.random.uniform(3, 8)),
            'aging_type': self.fading_params['aging_type'],
            'darken_strength': darken_strength,  # 使用上面计算的值
            'use_brown_overlay': self.fading_params['use_brown_overlay'],
            'overlay_opacity': self.fading_params['overlay_opacity']
        }

    def _synthesize_degradation(self, image: np.ndarray, params: dict) -> np.ndarray:
        """
        综合退化函数（简化版本，只包含核心效果）
        为了训练效率，这里只应用最重要的褪色效果

        注意：处理顺序很重要！
        - 色彩衰减会降低所有通道的值
        - 棕色叠加要在色彩衰减之后，否则效果会被削弱
        """
        result = image.copy().astype(np.float32)

        # 1. 应用褪色效果
        if params['aging_type'] in ['fade', 'both']:
            result = self._apply_fading(
                result,
                saturation=params['saturation'],
                brightness=params['brightness'],
                yellow=params['yellow'],
                sepia=params['sepia']
            )

        # 2. 应用变暗老化 (HSV部分，不包括棕色叠加)
        if params['aging_type'] in ['darken', 'both']:
            result = self._apply_darkening(
                result,
                darken_strength=params['darken_strength'],
                use_overlay=False,  # 先不应用棕色叠加
                overlay_opacity=params['overlay_opacity']
            )

        # 3. 非均匀色彩衰减 (会降低所有通道)
        result = self._apply_color_decay(result, params['decay_range'])

        # 4. 现在应用棕色叠加 (在色彩衰减之后，保持效果)
        if params['aging_type'] in ['darken', 'both'] and params['use_brown_overlay']:
            result = self._apply_brown_overlay_only(
                result,
                overlay_opacity=params['overlay_opacity']
            )

        # 5. 添加噪声
        if params['noise_level'] > 0:
            result = self._add_noise(result, params['noise_level'])

        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_fading(self, image: np.ndarray, saturation: float, brightness: float,
                     yellow: float, sepia: float) -> np.ndarray:
        """应用褪色效果"""
        img_float = image.astype(np.float32) / 255.0

        # 转换到HSV调整饱和度和亮度
        hsv = cv2.cvtColor(img_float, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] *= saturation
        hsv[:, :, 2] *= brightness
        hsv = np.clip(hsv, 0, 1)
        img_float = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 添加黄色调
        if yellow > 0:
            yellow_tint = np.array([0.2, 0.7, 1.0], dtype=np.float32)
            img_float = img_float * (1 - yellow) + yellow_tint * yellow

        # 添加棕褐色调
        if sepia > 0:
            sepia_kernel = np.array([
                [0.272, 0.534, 0.131],
                [0.349, 0.686, 0.168],
                [0.393, 0.769, 0.189]
            ], dtype=np.float32)
            img_rgb = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)
            sepia_img = cv2.transform(img_rgb, sepia_kernel)
            sepia_img = cv2.cvtColor(sepia_img, cv2.COLOR_RGB2BGR)
            img_float = img_float * (1 - sepia) + sepia_img * sepia

        return np.clip(img_float * 255, 0, 255)

    def _apply_brown_overlay_only(self, image: np.ndarray, overlay_opacity: float) -> np.ndarray:
        """
        只应用棕色叠加效果（不做HSV预处理）
        应该在色彩衰减之后调用，以保持效果

        Args:
            image: 输入图像，范围[0, 255]，float32
            overlay_opacity: 叠加不透明度

        Returns:
            应用棕色叠加后的图像
        """
        result = image.astype(np.float32)

        # 棕色叠加 (#50310f -> BGR: [15, 49, 80])
        brown_color = np.array([15, 49, 80], dtype=np.float32)
        brown_layer = np.ones_like(result) * brown_color

        # 直接应用，不降低不透明度
        result = result * (1 - overlay_opacity) + brown_layer * overlay_opacity

        return np.clip(result, 0, 255)

    def _apply_darkening(self, image: np.ndarray, darken_strength: float,
                        use_overlay: bool, overlay_opacity: float) -> np.ndarray:
        """应用变暗老化"""
        result = image.astype(np.float32)

        if use_overlay:
            # 组合模式：HSV + 棕色叠加
            hsv = cv2.cvtColor(result / 255.0, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] *= (1 - darken_strength * 0.4)
            hsv[:, :, 1] *= 1.2
            hsv = np.clip(hsv, 0, 1)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) * 255

            # 棕色叠加
            brown_color = np.array([15, 49, 80], dtype=np.float32)
            brown_layer = np.ones_like(result) * brown_color
            adjusted_opacity = overlay_opacity * 0.85
            result = result * (1 - adjusted_opacity) + brown_layer * adjusted_opacity
        else:
            # 纯HSV方法
            hsv = cv2.cvtColor(result / 255.0, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] *= (1 - darken_strength)
            hsv[:, :, 1] *= 1.4
            hsv = np.clip(hsv, 0, 1)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR) * 255

        return np.clip(result, 0, 255)

    def _apply_color_decay(self, image: np.ndarray, decay_range: Tuple[float, float]) -> np.ndarray:
        """应用非均匀色彩衰减"""
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

    def _add_noise(self, image: np.ndarray, noise_level: int) -> np.ndarray:
        """添加噪声"""
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        return np.clip(noisy_image, 0, 255)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        获取一个数据样本

        Returns:
            gt: 高质量图片，RGB，范围[-1, 1]，float32
            lq: 低质量（褪色）图片，RGB，范围[0, 1]，float32
            prompt: 文本描述
        """
        filename = self.image_files[index]

        # 加载原始图片
        img_gt = self.load_and_crop_image(filename)

        # 应用褪色效果生成低质量图片
        img_lq = self.apply_fading_degradation(img_gt)

        # 获取prompt
        prompt = self.prompts[filename]

        # 数据格式转换
        # GT: RGB, [0, 255] -> RGB, [-1, 1]
        gt = (img_gt.astype(np.float32) / 255.0 * 2 - 1).astype(np.float32)

        # LQ: RGB, [0, 255] -> RGB, [0, 1]
        lq = (img_lq.astype(np.float32) / 255.0).astype(np.float32)

        return gt, lq, prompt

    def __len__(self) -> int:
        return len(self.image_files)
