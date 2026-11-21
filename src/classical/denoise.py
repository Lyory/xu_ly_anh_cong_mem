import cv2
import numpy as np
from scipy.ndimage import uniform_filter

METHODS = ("mean", "gaussian", "median", "bilateral", "nlm", "wiener")


def denoise(img, method="bilateral", size=3):
    if size < 3 or size % 2 != 1:
        raise ValueError("Filter size must be odd and >= 3")
    if method == "mean":
        return cv2.blur(img, (size, size))
    if method == "gaussian":
        return cv2.GaussianBlur(img, (size, size), .8)
    if method == "median":
        return cv2.medianBlur(img, size)
    if method == "bilateral":
        return cv2.bilateralFilter(img, 5, 25, 3)
    if method == "nlm":
        return cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)
    if method == "wiener":
        # Adaptive spatial Wiener; estimate local noise power, per channel.
        source = img.astype(np.float32)
        mean = uniform_filter(source, (size, size, 1))
        variance = np.maximum(uniform_filter(source**2, (size, size, 1))-mean**2, 0)
        noise = np.mean(variance, axis=(0, 1), keepdims=True)
        gain = np.maximum(variance-noise, 0) / np.maximum(variance, 1e-6)
        return np.clip(mean + gain*(source-mean), 0, 255).astype(np.uint8)
    raise ValueError(f"Unknown denoiser: {method}")
