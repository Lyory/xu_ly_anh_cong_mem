from pathlib import Path
import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def image_files(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Cannot write image: {path}")


def validate_rgb(image):
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected H x W x 3 RGB uint8 image")
    if min(image.shape[:2]) < 16:
        raise ValueError("Image dimensions must be at least 16 pixels")


def identity_grid(height, width):
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    return np.stack([x, y], axis=-1).astype(np.float32)


def sample_grid(image, grid):
    h, w = image.shape[:2]
    return cv2.remap(image, ((grid[..., 0] + 1) * (w - 1) / 2).astype(np.float32),
                     ((grid[..., 1] + 1) * (h - 1) / 2).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
