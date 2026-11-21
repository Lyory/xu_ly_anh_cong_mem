import cv2
import numpy as np
from src.classical.edge_detection import detect_edges


def rotate(img, correction):
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), correction, 1)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def estimate_skew(img, method="hough", max_angle=12):
    """Return OpenCV correction angle (positive counterclockwise), in degrees."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    if method == "hough":
        lines = cv2.HoughLinesP(detect_edges(img), 1, np.pi/720, threshold=30,
                               minLineLength=max(20, img.shape[1]//8), maxLineGap=15)
        angles = []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                angle = np.degrees(np.arctan2(y2-y1, x2-x1))
                angle = (angle+90) % 180 - 90
                if abs(angle) <= max_angle:
                    angles.append(angle)
        return float(np.median(angles)) if angles else 0.
    if method == "projection":
        scale = min(1, 600/max(gray.shape))
        small = cv2.resize(mask, None, fx=scale, fy=scale)
        scores = []
        candidates = np.arange(-max_angle, max_angle+.25, .5)
        h, w = small.shape
        for angle in candidates:
            matrix = cv2.getRotationMatrix2D(((w-1)/2, (h-1)/2), angle, 1)
            rotated = cv2.warpAffine(small, matrix, (w, h), borderValue=0)
            scores.append(np.var(np.sum(rotated.astype(np.float32), axis=1)))
        return float(candidates[np.argmax(scores)]) if np.any(mask) else 0.
    if method == "min_area_rect":
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        y, x = np.nonzero(mask)
        if len(x) < 10:
            return 0.
        angle = cv2.minAreaRect(np.column_stack([x, y]).astype(np.float32))[-1]
        angle = (angle+45) % 90 - 45
        return float(angle) if abs(angle) <= max_angle else 0.
    raise ValueError(f"Unknown deskew method: {method}")


def deskew(img, method="hough"):
    angle = estimate_skew(img, method)
    return rotate(img, angle) if abs(angle) > .15 else img.copy(), angle
