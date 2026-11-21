from PIL import Image, ImageOps
import cv2
import numpy as np
from src.utils import validate_rgb


def load_image(path):
    """Đọc ảnh RGB và xoay đúng theo hướng chụp ghi trong EXIF."""
    with Image.open(path) as image:
        return np.array(ImageOps.exif_transpose(image).convert("RGB"))


def gray_world(img):
    """Cân bằng trắng theo tỉ số giữa trung bình chung và trung bình từng kênh."""
    values = img.astype(np.float32)
    means = values.mean(axis=(0, 1))
    gains = np.clip(means.mean() / np.maximum(means, 1), .75, 1.33)
    return np.clip(values*gains, 0, 255).astype(np.uint8)


def preprocess(img, max_size=1800, white_balance=False, padding=0):
    """Giới hạn cạnh dài, giữ tỉ lệ, cân bằng trắng và thêm lề nếu được chọn."""
    validate_rgb(img)
    if max_size is not None and max_size < 16:
        raise ValueError("max_size must be >= 16")
    if padding < 0:
        raise ValueError("padding must be nonnegative")
    output = img.copy()
    if max_size and max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        output = cv2.resize(output, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if white_balance:
        output = gray_world(output)
    if padding:
        output = cv2.copyMakeBorder(output, padding, padding, padding, padding,
                                   cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return output
