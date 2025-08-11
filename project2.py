import hashlib
import os
from typing import Tuple, Optiona, Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
from scipy.fftpack import dct, idct


class WatermarkDetector:
    def __init__(self, secret_key: str = "default_key", alpha: float = 0.05):
        # 初始化水印检测器
        self.secret_key = secret_key
        self.alpha = alpha  # 水印强度，值越大鲁棒性越强但可见性越高
        self.rng = np.random.RandomState(int(hashlib.md5(secret_key.encode()).hexdigest(), 16))

    def _generate_watermark(self, shape: Tuple[int, int], bits: int = 256) -> np.ndarray:
        # 生成伪随机水印图案
        watermark = np.zeros(shape, dtype=np.float32)
        # 选择非直流分量区域嵌入水印
        valid_positions = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                if i > 5 and j > 5:  # 避开低频区域
                    valid_positions.append((i, j))

        # 随机选择位置
        selected = self.rng.choice(len(valid_positions), bits, replace=False)
        for idx in selected:
            i, j = valid_positions[idx]
            watermark[i, j] = 1 if self.rng.rand() > 0.5 else -1

        return watermark

    def embed_watermark(self, img_path: str, output_path: str, bits: int = 256, method: str = 'dct') -> None:
        """
        嵌入水印到图片
        :param img_path: 原始图片路径
        :param output_path: 含水印图片输出路径
        :param bits: 水印位数
        :param method: 嵌入方法，'dct'或'lsb'
        """
        # 读取图片并转为灰度图
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图片: {img_path}")

        # 转为灰度图处理
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 根据选择的方法嵌入水印
        if method == 'dct':
            watermarked_gray = self._embed_dct(gray_img, bits)
        elif method == 'lsb':
            watermarked_gray = self._embed_lsb(gray_img, bits)
        else:
            raise ValueError("支持的嵌入方法: 'dct' 或 'lsb'")

        # 转换回BGR格式保存
        watermarked_img = cv2.cvtColor(watermarked_gray, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(output_path, watermarked_img)
        print(f"水印已嵌入并保存至: {output_path}")

    def _embed_dct(self, img: np.ndarray, bits: int) -> np.ndarray:
        # 在DCT域嵌入水印
        h, w = img.shape
        # 确保尺寸是8的倍数（DCT处理块大小）
        h = h - (h % 8)
        w = w - (w % 8)
        img = img[:h, :w].astype(np.float32)

        # 生成水印
        watermark = self._generate_watermark((h, w), bits)

        # 分块处理
        watermarked = np.copy(img)
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                # 对8x8块进行DCT变换
                block = img[i:i + 8, j:j + 8]
                dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

                # 在中频区域嵌入水印（选择(3,3)位置）
                dct_block[3, 3] += self.alpha * watermark[i, j]

                # 逆DCT变换
                idct_block = idct(idct(dct_block, axis=1, norm='ortho'), axis=0, norm='ortho')
                watermarked[i:i + 8, j:j + 8] = idct_block

        return np.clip(watermarked, 0, 255).astype(np.uint8)

    def _embed_lsb(self, img: np.ndarray, bits: int) -> np.ndarray:
        # 在LSB（最低有效位）嵌入水印
        h, w = img.shape
        watermark = self._generate_watermark((h, w), bits)

        # 生成嵌入位置
        positions = []
        for i in range(h):
            for j in range(w):
                if watermark[i, j] != 0:
                    positions.append((i, j))

        # 打乱位置顺序
        self.rng.shuffle(positions)
        positions = positions[:bits]

        # 修改最低有效位
        watermarked = np.copy(img)
        for i, j in positions:
            bit = 1 if watermark[i, j] > 0 else 0
            watermarked[i, j] = (watermarked[i, j] & 0xFE) | bit  # 保留高7位，修改最低位

        return watermarked.astype(np.uint8)

    def extract_watermark(self, watermarked_path: str, original_path: Optional[str] = None,
                          bits: int = 256, method: str = 'dct') -> np.ndarray:
        """
        从图片中提取水印
        :param watermarked_path: 含水印图片路径
        :param original_path: 原始图片路径（非盲提取时使用）
        :param bits: 水印位数
        :param method: 提取方法，需与嵌入方法一致
        """
        # 读取图片
        watermarked_img = cv2.imread(watermarked_path, cv2.IMREAD_GRAYSCALE)
        if watermarked_img is None:
            raise FileNotFoundError(f"无法读取图片: {watermarked_path}")

        original_img = None
        if original_path:
            original_img = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
            if original_img is None:
                raise FileNotFoundError(f"无法读取原始图片: {original_path}")
            # 确保尺寸匹配
            h, w = min(watermarked_img.shape[0], original_img.shape[0]), min(watermarked_img.shape[1],
                                                                             original_img.shape[1])
            watermarked_img = watermarked_img[:h, :w]
            original_img = original_img[:h, :w]

        # 根据方法提取水印
        if method == 'dct':
            return self._extract_dct(watermarked_img, original_img, bits)
        elif method == 'lsb':
            return self._extract_lsb(watermarked_img, bits)
        else:
            raise ValueError("支持的提取方法: 'dct' 或 'lsb'")

    def _extract_dct(self, watermarked: np.ndarray, original: Optional[np.ndarray], bits: int) -> np.ndarray:
        # 从DCT域提取水印
        h, w = watermarked.shape
        h = h - (h % 8)
        w = w - (w % 8)
        watermarked = watermarked[:h, :w].astype(np.float32)

        # 生成水印模板（用于盲提取）
        watermark_shape = (h, w)
        watermark_template = self._generate_watermark(watermark_shape, bits)

        extracted = np.zeros(watermark_shape, dtype=np.float32)

        if original is not None:
            # 非盲提取：对比原始图像和含水印图像
            original = original[:h, :w].astype(np.float32)
            for i in range(0, h, 8):
                for j in range(0, w, 8):
                    orig_block = original[i:i + 8, j:j + 8]
                    watermarked_block = watermarked[i:i + 8, j:j + 8]

                    # 计算DCT
                    orig_dct = dct(dct(orig_block, axis=0, norm='ortho'), axis=1, norm='ortho')
                    watermarked_dct = dct(dct(watermarked_block, axis=0, norm='ortho'), axis=1, norm='ortho')

                    # 提取水印信息
                    extracted[i, j] = (watermarked_dct[3, 3] - orig_dct[3, 3]) / self.alpha
        else:
            # 盲提取：仅使用含水印图像
            for i in range(0, h, 8):
                for j in range(0, w, 8):
                    if watermark_template[i, j] != 0:  # 只提取有水印的位置
                        block = watermarked[i:i + 8, j:j + 8]
                        dct_block = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
                        extracted[i, j] = dct_block[3, 3]

        return extracted

    def _extract_lsb(self, watermarked: np.ndarray, bits: int) -> np.ndarray:
        # 从LSB提取水印
        h, w = watermarked.shape
        watermark_template = self._generate_watermark((h, w), bits)

        extracted = np.zeros((h, w), dtype=np.float32)
        # 提取最低有效位
        for i in range(h):
            for j in range(w):
                if watermark_template[i, j] != 0:
                    bit = watermarked[i, j] & 1  # 获取最低位
                    extracted[i, j] = 1 if bit == 1 else -1

        return extracted

    def calculate_similarity(self, original_watermark: np.ndarray, extracted_watermark: np.ndarray) -> float:
        # 只考虑有效水印位置
        mask = (original_watermark != 0)
        orig = original_watermark[mask]
        ext = extracted_watermark[mask]

        if len(orig) == 0 or len(ext) == 0:
            return 0.0

        # 计算归一化相关系数
        mean_orig = np.mean(orig)
        mean_ext = np.mean(ext)

        numerator = np.sum((orig - mean_orig) * (ext - mean_ext))
        denominator = np.sqrt(np.sum((orig - mean_orig) ** 2) * np.sum((ext - mean_ext) ** 2))

        return numerator / denominator if denominator != 0 else 0.0

    def apply_attack(self, img_path: str, attack_type: str, output_path: str, **kwargs) -> None:
        """
        对图片应用攻击
        :param img_path: 图片路径
        :param attack_type: 攻击类型
        :param output_path: 攻击后图片保存路径
        """
        img = Image.open(img_path).convert('RGB')
        attacked_img = None

        if attack_type == 'flip':
            # 翻转
            flip_direction = kwargs.get('direction', 'horizontal')
            if flip_direction == 'horizontal':
                attacked_img = img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                attacked_img = img.transpose(Image.FLIP_TOP_BOTTOM)

        elif attack_type == 'rotate':
            # 旋转
            angle = kwargs.get('angle', 30)
            attacked_img = img.rotate(angle, expand=True)

        elif attack_type == 'crop':
            # 裁剪
            ratio = kwargs.get('ratio', 0.2)
            w, h = img.size
            crop_w = int(w * ratio)
            crop_h = int(h * ratio)
            attacked_img = img.crop((crop_w, crop_h, w - crop_w, h - crop_h))
            # 缩放回原尺寸
            attacked_img = attacked_img.resize((w, h))

        elif attack_type == 'resize':
            # 缩放
            scale = kwargs.get('scale', 0.5)
            w, h = img.size
            new_size = (int(w * scale), int(h * scale))
            attacked_img = img.resize(new_size)
            # 缩放回原尺寸
            attacked_img = attacked_img.resize((w, h))

        elif attack_type == 'brightness':
            # 亮度调整
            factor = kwargs.get('factor', 1.5)
            enhancer = ImageEnhance.Brightness(img)
            attacked_img = enhancer.enhance(factor)

        elif attack_type == 'contrast':
            # 对比度调整
            factor = kwargs.get('factor', 1.5)
            enhancer = ImageEnhance.Contrast(img)
            attacked_img = enhancer.enhance(factor)

        elif attack_type == 'noise':
            # 添加高斯噪声
            sigma = kwargs.get('sigma', 10)
            img_np = np.array(img)
            noise = np.random.normal(0, sigma, img_np.shape).astype(np.int16)
            attacked_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            attacked_img = Image.fromarray(attacked_np)

        elif attack_type == 'jpeg':
            # JPEG压缩
            quality = kwargs.get('quality', 50)
            attacked_img = img
            attacked_img.save(output_path, 'JPEG', quality=quality)
            return  # 直接保存

        else:
            raise ValueError(f"不支持的攻击类型: {attack_type}")

        if attacked_img:
            attacked_img.save(output_path)
            print(f"攻击后图片已保存至: {output_path}")

    def robustness_test(self, original_img: str, watermarked_img: str, output_dir: str = "attack_results",
                        method: str = 'dct', bits: int = 256) -> dict:
        """
        进行鲁棒性测试
        :return: 各种攻击下的水印相似度
        """
        os.makedirs(output_dir, exist_ok=True)

        # 生成原始水印用于对比
        test_img = cv2.imread(original_img, cv2.IMREAD_GRAYSCALE)
        h, w = test_img.shape
        original_watermark = self._generate_watermark((h, w), bits)

        # 无攻击情况下的提取结果
        extracted_original = self.extract_watermark(watermarked_img, original_img, bits, method)
        base_similarity = self.calculate_similarity(original_watermark, extracted_original)

        results = {
            "original": base_similarity
        }

        # 定义要测试的攻击
        attacks = [
            ('flip', {'direction': 'horizontal'}),
            ('flip', {'direction': 'vertical'}),
            ('rotate', {'angle': 15}),
            ('rotate', {'angle': 30}),
            ('crop', {'ratio': 0.1}),
            ('crop', {'ratio': 0.2}),
            ('resize', {'scale': 0.5}),
            ('resize', {'scale': 1.5}),
            ('brightness', {'factor': 0.5}),
            ('brightness', {'factor': 1.5}),
            ('contrast', {'factor': 0.5}),
            ('contrast', {'factor': 1.5}),
            ('noise', {'sigma': 5}),
            ('noise', {'sigma': 10}),
            ('jpeg', {'quality': 70}),
            ('jpeg', {'quality': 30}),
        ]

        # 对每种攻击进行测试
        for attack_type, params in attacks:
            attack_name = f"{attack_type}_{'_'.join([f'{k}{v}' for k, v in params.items()])}"
            attack_path = os.path.join(output_dir, f"{attack_name}.png")

            # 应用攻击
            self.apply_attack(watermarked_img, attack_type, attack_path, **params)

            # 提取水印
            extracted = self.extract_watermark(attack_path, original_img, bits, method)

            # 计算相似度
            similarity = self.calculate_similarity(original_watermark, extracted)
            results[attack_name] = similarity

            print(f"{attack_name} - 相似度: {similarity:.4f}")

        # 可视化结果
        self._visualize_results(original_img, watermarked_img, original_watermark,
                                extracted_original, results, output_dir)

        return results

    def _visualize_results(self, original_path: str, watermarked_path: str, original_wm: np.ndarray,
                           extracted_wm: np.ndarray, results: dict, output_dir: str):
        """可视化测试结果"""
        plt.figure(figsize=(15, 10))

        # 原始图片
        plt.subplot(221)
        img = cv2.imread(original_path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('原始图片')
        plt.axis('off')

        # 含水印图片
        plt.subplot(222)
        img = cv2.imread(watermarked_path)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('含水印图片')
        plt.axis('off')

        # 原始水印
        plt.subplot(223)
        plt.imshow(original_wm, cmap='gray')
        plt.title('原始水印')
        plt.axis('off')

        # 提取的水印
        plt.subplot(224)
        plt.imshow(extracted_wm, cmap='gray')
        plt.title(f'提取的水印 (相似度: {results["original"]:.4f})')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'watermark_visualization.png'))
        plt.close()

        # 绘制鲁棒性测试结果
        plt.figure(figsize=(12, 8))
        attacks = list(results.keys())[1:]  # 排除原始情况
        similarities = [results[attack] for attack in attacks]

        plt.barh(attacks, similarities, color='skyblue')
        plt.axvline(x=0.5, color='r', linestyle='--', label='相似度阈值 (0.5)')
        plt.xlabel('水印相似度')
        plt.title('不同攻击下的水印鲁棒性测试结果')
        plt.xlim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'robustness_results.png'))
        plt.close()


# 示例使用
if __name__ == "__main__":
    # 初始化水印检测器
    detector = WatermarkDetector(secret_key="my_secure_key", alpha=0.08)

    # 配置文件路径
    original_image = "test_image.jpg"  # 替换为你的图片路径
    watermarked_image = "watermarked_image.jpg"
    attack_dir = "attack_results"

    # 嵌入水印
    detector.embed_watermark(original_image, watermarked_image, bits=512, method='dct')

    # 进行鲁棒性测试
    print("开始鲁棒性测试...")
    results = detector.robustness_test(original_image, watermarked_image, attack_dir, method='dct', bits=512)

    # 输出测试结果摘要
    print("\n鲁棒性测试结果摘要:")
    print(f"原始提取相似度: {results['original']:.4f}")
    print("\n最具破坏性的攻击:")
    sorted_attacks = sorted(results.items(), key=lambda x: x[1])[:5]
    for attack, sim in sorted_attacks:
        if attack != 'original':
            print(f"{attack}: {sim:.4f}")

    print("\n测试完成，结果已保存至 attack_results 目录")