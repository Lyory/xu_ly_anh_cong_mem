"""Image metrics assume spatially aligned RGB images, range [0,255]."""
import numpy as np
from scipy.ndimage import gaussian_filter


def _pair(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError("Metrics require aligned images of identical shape; do not resize ground truth")
    return prediction.astype(np.float64) / 255, target.astype(np.float64) / 255


def psnr(prediction, target):
    x, y = _pair(prediction, target)
    mse = np.mean((x-y)**2)
    return float('inf') if mse == 0 else float(-10 * np.log10(mse))


def _ssim_components(x, y):
    sigma = (1.5, 1.5, 0) if x.ndim == 3 else 1.5
    ux, uy = gaussian_filter(x, sigma), gaussian_filter(y, sigma)
    vx = np.maximum(gaussian_filter(x*x, sigma)-ux*ux, 0)
    vy = np.maximum(gaussian_filter(y*y, sigma)-uy*uy, 0)
    covariance = gaussian_filter(x*y, sigma)-ux*uy
    luminance = (2*ux*uy+.01**2)/(ux*ux+uy*uy+.01**2)
    contrast = (2*covariance+.03**2)/(vx+vy+.03**2)
    return float(np.mean(luminance * contrast)), float(np.mean(contrast))


def ssim(prediction, target):
    return _ssim_components(*_pair(prediction, target))[0]


def ms_ssim(prediction, target):
    """Five-scale SSIM, Gaussian population moments, reflect borders.

    Small images use fewer scales and renormalized weights. This documented
    implementation can differ slightly from valid-window benchmark packages.
    """
    x, y = _pair(prediction, target)
    weights = np.array([.0448, .2856, .3001, .2363, .1333])
    contrasts, values = [], []
    for _ in weights:
        value, contrast = _ssim_components(x, y)
        values.append(value)
        contrasts.append(contrast)
        if min(x.shape[:2]) < 22:
            break
        x = gaussian_filter(x, (1, 1, 0))[::2, ::2]
        y = gaussian_filter(y, (1, 1, 0))[::2, ::2]
    weights = weights[:len(values)]
    weights /= weights.sum()
    terms = np.maximum(contrasts[:-1] + [values[-1]], 1e-8)
    return float(np.prod(terms**weights))


def local_distortion(predicted_bmap, target_bmap, image_shape=None):
    """Dense map endpoint error in pixels (LD proxy, NOT official DocUNet LD).

    Requires two actual normalized backward maps; cannot infer GT from photos.
    """
    if predicted_bmap.shape != target_bmap.shape or predicted_bmap.shape[-1] != 2:
        raise ValueError("Expected matching HxWx2 backward maps")
    h, w = image_shape[:2] if image_shape is not None else target_bmap.shape[:2]
    error = (predicted_bmap-target_bmap) * np.array([(w-1)/2, (h-1)/2])
    return float(np.linalg.norm(error, axis=-1).mean())


def text_metrics(prediction, reference):
    from jiwer import cer, wer
    # Empty-reference behavior is explicit and stable across jiwer versions.
    if not reference.strip():
        return {"cer": 0. if not prediction.strip() else float(len(prediction)),
                "wer": float(len(prediction.split()))}
    return {"cer": float(cer(reference, prediction)), "wer": float(wer(reference, prediction))}


def image_metrics(prediction, target):
    return {"psnr": psnr(prediction, target), "ssim": ssim(prediction, target),
            "ms_ssim": ms_ssim(prediction, target)}
