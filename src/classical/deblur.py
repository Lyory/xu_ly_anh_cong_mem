import cv2
import numpy as np
from scipy.signal import fftconvolve


def unsharp_mask(image, amount, sigma=1.1):
    """Làm nét theo I + a × (I − Gσ * I), với Gσ * I là ảnh đã làm mượt."""
    smooth = cv2.GaussianBlur(image, (0, 0), sigma)
    pixels = image.astype(np.float32)
    sharpened = pixels + amount * (pixels - smooth)
    return np.clip(np.rint(sharpened), 0, 255).astype(np.uint8)


def gaussian_psf(size=5, sigma=1.2):
    """Tạo hàm lan truyền điểm Gaussian hai chiều từ tích của hai nhân một chiều."""
    kernel = cv2.getGaussianKernel(size, sigma)
    return kernel @ kernel.T


def deblur(img, method="unsharp", psf=None, iterations=12, balance=.02):
    """Khử mờ bằng mặt nạ làm nét, bộ lọc Wiener hoặc lặp Richardson–Lucy."""
    if method == "unsharp":
        return unsharp_mask(img, amount=.6)
    if method not in {"wiener", "richardson_lucy"}:
        raise ValueError(f"Unknown deblur method: {method}")
    psf = gaussian_psf() if psf is None else np.asarray(psf, np.float32)
    if psf.ndim != 2 or psf.sum() <= 0 or balance <= 0 or iterations < 1:
        raise ValueError("Invalid PSF, balance or iterations")
    psf = psf / psf.sum()
    source = np.pad(img.astype(np.float32) / 255, ((16, 16), (16, 16), (0, 0)), mode="reflect")
    result = np.empty_like(source)
    for ch in range(3):
        plane = source[..., ch]
        if method == "wiener":
            kernel = np.zeros(plane.shape, np.float32)
            kh, kw = psf.shape
            kernel[:kh, :kw] = psf
            kernel = np.roll(kernel, (-(kh//2), -(kw//2)), axis=(0, 1))
            otf = np.fft.fft2(kernel)
            estimate = np.fft.ifft2(np.fft.fft2(plane) * otf.conj() / (np.abs(otf)**2 + balance)).real
        else:
            estimate = np.maximum(plane, 1e-4)
            for _ in range(iterations):
                relative = plane / np.maximum(fftconvolve(estimate, psf, mode="same"), 1e-5)
                estimate *= fftconvolve(relative, psf[::-1, ::-1], mode="same")
        result[..., ch] = estimate
    return (np.clip(result[16:-16, 16:-16], 0, 1) * 255).astype(np.uint8)
