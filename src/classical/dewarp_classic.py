"""Conservative baseline: fit curved text components, interpolate vertical offsets."""
import cv2
import numpy as np
from scipy.interpolate import interp1d


def textline_curves(img, degree=2):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 12)
    h, w = gray.shape
    # Horizontal closing joins characters while preserving separated text lines.
    joined = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                             np.ones((3, max(9, w // 35)), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
    curves = []
    for label in range(1, n):
        x, y, width, height, area = stats[label]
        if width < .3*w or height > .15*h or area < 50:
            continue
        xs, ys = [], []
        for col in range(x, x+width, max(2, w//160)):
            rows = np.flatnonzero((labels[:, col] == label) & (binary[:, col] > 0))
            if len(rows):
                xs.append(col / max(w-1, 1))
                ys.append(np.median(rows))
        if len(xs) < 8:
            continue
        coefficients = np.polyfit(xs, ys, degree)
        residual = np.asarray(ys) - np.polyval(coefficients, xs)
        if np.median(np.abs(residual)) > max(2., height * .25):
            continue
        curve = np.polyval(coefficients, np.linspace(0, 1, w))
        # Preserve the linear slope; rectify curvature only.
        line = np.polyval(np.polyfit(xs, np.polyval(coefficients, xs), 1), np.linspace(0, 1, w))
        offset = curve - line
        if np.max(np.abs(offset)) < .08*h:
            curves.append((float(np.median(line)), offset.astype(np.float32)))
    return sorted(curves, key=lambda item: item[0])


def dewarp_classic(img, degree=2, return_map=False):
    h, w = img.shape[:2]
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    curves = textline_curves(img, degree)
    if len(curves) >= 3:
        positions = [0.] + [p for p, _ in curves] + [float(h-1)]
        offsets = np.stack([np.zeros(w)] + [offset for _, offset in curves] + [np.zeros(w)])
        # Merge duplicate line heights before interpolating.
        positions, indices = np.unique(positions, return_index=True)
        correction = interp1d(positions, offsets[indices], axis=0, bounds_error=False,
                              fill_value=0)(np.arange(h)).astype(np.float32)
        # Avoid folded vertical sampling grids on uncertain detections.
        if np.min(np.diff(y + correction, axis=0)) > .2:
            y += correction
    grid = np.stack([x * 2/(w-1)-1, y * 2/(h-1)-1], -1)
    output = cv2.remap(img, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return (output, grid.astype(np.float32)) if return_map else output
