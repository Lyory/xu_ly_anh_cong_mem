import cv2
import numpy as np
from src.classical.frequency import homomorphic_luminance


def estimate_background(img, kernel_size=None):
    h, w = img.shape[:2]
    size = kernel_size or max(7, int(min(h, w) * .035) | 1)
    size = max(3, min(size | 1, (min(h, w) - 1) | 1))
    dilated = cv2.dilate(img, np.ones((5, 5), np.uint8))
    return cv2.medianBlur(dilated, size)


def remove_shadow(img, method="divide"):
    if method == "divide":
        background = estimate_background(img).astype(np.float32)
        return np.clip(img.astype(np.float32) / np.maximum(background, 8) * 255, 0, 255).astype(np.uint8)
    if method != "homomorphic":
        raise ValueError(f"Unknown shadow method: {method}")
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = homomorphic_luminance(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
