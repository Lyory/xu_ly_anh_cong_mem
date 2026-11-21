"""Seedable synthetic degradations. A bmap samples warped INPUT into clean OUTPUT."""
import cv2
import numpy as np
from src.utils import identity_grid, sample_grid


def add_shadow(img, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    illumination = np.ones((h, w), np.float32)
    for _ in range(int(rng.integers(1, 4))):
        mask = np.zeros((h, w), np.float32)
        polygon = (rng.random((5, 2)) * [w, h]).astype(np.int32)
        cv2.fillConvexPoly(mask, cv2.convexHull(polygon), 1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(2, min(h, w) * rng.uniform(.03, .15)))
        illumination *= 1 - rng.uniform(.2, .65) * mask
    angle = rng.uniform(0, 2 * np.pi)
    gradient = np.cos(angle) * x / w + np.sin(angle) * y / h
    gradient = (gradient - gradient.min()) / max(float(np.ptp(gradient)), 1e-6)
    illumination *= 1 - rng.uniform(.1, .4) * gradient
    return np.clip(img.astype(np.float32) * illumination[..., None], 0, 255).astype(np.uint8)


def add_blur(img, rng=None, kind=None):
    rng = np.random.default_rng() if rng is None else rng
    kind = rng.choice(["gaussian", "motion", "defocus"]) if kind is None else kind
    size = int(rng.choice([3, 5, 7, 9]))
    if kind == "gaussian":
        return cv2.GaussianBlur(img, (size, size), rng.uniform(.6, 2.2))
    kernel = np.zeros((size, size), np.float32)
    if kind == "motion":
        kernel[size // 2, :] = 1
        matrix = cv2.getRotationMatrix2D((size // 2, size // 2), rng.uniform(0, 180), 1)
        kernel = cv2.warpAffine(kernel, matrix, (size, size))
    elif kind == "defocus":
        cv2.circle(kernel, (size // 2, size // 2), max(1, size // 2 - 1), 1, -1)
    else:
        raise ValueError(f"Unknown blur: {kind}")
    return cv2.filter2D(img, -1, kernel / kernel.sum())


def add_noise_jpeg(img, rng=None, salt_pepper=0.):
    rng = np.random.default_rng() if rng is None else rng
    noisy = img.astype(np.float32) * rng.uniform(.90, 1.08, (1, 1, 3))
    noisy += rng.normal(0, rng.uniform(1, 7), img.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    if salt_pepper:
        if not 0 <= salt_pepper <= 1:
            raise ValueError("salt_pepper must lie in [0, 1]")
        mask = rng.random(img.shape[:2])
        noisy[mask < salt_pepper/2] = 0
        noisy[mask > 1-salt_pepper/2] = 255
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(50, 96))])
    if not ok:
        raise RuntimeError("JPEG encoder failed")
    return cv2.cvtColor(cv2.imdecode(buffer, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def add_warp(img, rng=None, strength=.07, perspective=True):
    """Return warped RGB and HxWx2 normalized backward map, align_corners=True.

    Build a smooth clean->camera mapping. Invert it to synthesize the camera
    image; supervise the original mapping to restore clean coordinates.
    Small bounded gradients keep this synthetic surface invertible.
    """
    if not 0 <= strength <= .12:
        raise ValueError("strength must lie in [0, .12] to preserve invertibility")
    rng = np.random.default_rng() if rng is None else rng
    h, w = img.shape[:2]
    base = identity_grid(h, w)
    a, b, c = rng.uniform(-strength, strength, 3)

    def displacement(q):
        x, y = q[..., 0], q[..., 1]
        envelope = (1 - x*x) * (1 - y*y)
        return np.stack([a * np.sin(np.pi * y) * envelope,
                         (b * np.sin(np.pi * x) + c * (1-x*x)) * envelope], -1)

    corners = np.float32([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    destination = corners * rng.uniform(.91, .99, (4, 2)).astype(np.float32) if perspective else corners
    matrix = cv2.getPerspectiveTransform(corners, destination)
    forward = cv2.perspectiveTransform((base + displacement(base)).reshape(1, -1, 2), matrix).reshape(h, w, 2)
    planar = cv2.perspectiveTransform(base.reshape(1, -1, 2), np.linalg.inv(matrix)).reshape(h, w, 2)
    inverse = planar.copy()
    for _ in range(30):
        inverse = planar - displacement(np.clip(inverse, -1, 1))
    warped = sample_grid(img, inverse.astype(np.float32))
    return warped, forward.astype(np.float32)


def paste_on_background(img, bg, rng=None, return_mask=False):
    """Place a page in an inset rectangle. Do not use without composing bmaps."""
    rng = np.random.default_rng() if rng is None else rng
    h, w = img.shape[:2]
    canvas = cv2.resize(bg, (w, h)).copy()
    margin = float(rng.uniform(.04, .12))
    dx, dy = max(1, int(w * margin)), max(1, int(h * margin))
    canvas[dy:h-dy, dx:w-dx] = cv2.resize(img, (w-2*dx, h-2*dy))
    mask = np.zeros((h, w), np.uint8)
    mask[dy:h-dy, dx:w-dx] = 255
    return (canvas, mask) if return_mask else canvas
