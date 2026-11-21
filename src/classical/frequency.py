"""FFT diagnostics and an explicitly heuristic motion PSF estimate."""
import cv2
import numpy as np


def fft_spectrum(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    window = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2((gray-gray.mean())*window))))


def motion_psf(length=7, angle=0):
    size = max(3, int(length) | 1)
    kernel = np.zeros((size, size), np.float32)
    kernel[size//2, :] = 1
    matrix = cv2.getRotationMatrix2D((size//2, size//2), -angle, 1)
    kernel = cv2.warpAffine(kernel, matrix, (size, size))
    return kernel / max(float(kernel.sum()), 1e-6)


def estimate_motion_psf(img, length=7):
    """Find a low-energy radial FFT direction; length is supplied, not inferred.

    Text orientation can dominate this estimate. Validate against known PSFs.
    """
    spectrum = fft_spectrum(img)
    h, w = spectrum.shape
    radii = np.linspace(.12, .42, 80) * min(h, w)
    angles = np.arange(180)
    scores = []
    for angle in angles:
        radians = np.deg2rad(angle)
        x = np.clip(np.rint(w//2 + radii*np.cos(radians)), 0, w-1).astype(int)
        y = np.clip(np.rint(h//2 + radii*np.sin(radians)), 0, h-1).astype(int)
        scores.append(float(spectrum[y, x].mean()))
    angle = float(angles[np.argmin(scores)])
    return motion_psf(length, angle), {"angle": angle, "length": length, "heuristic": True}


def homomorphic_luminance(luminance, cutoff=.025):
    values = np.log1p(luminance.astype(np.float32))
    h, w = values.shape
    fy, fx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing="ij")
    highpass = 1-np.exp(-(fx*fx+fy*fy)/(2*cutoff**2))
    result = np.fft.ifft2(np.fft.fft2(values)*highpass).real
    return np.clip(np.expm1(result+np.percentile(values, 90)), 0, 255).astype(np.uint8)
