import cv2
import numpy as np

METHODS = ("sobel", "prewitt", "scharr", "laplacian", "log", "canny")


def gray_float(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)


def gradients(img, method="sobel"):
    gray = gray_float(img)
    if method == "sobel":
        return cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=1/8), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=1/8)
    if method == "scharr":
        return cv2.Scharr(gray, cv2.CV_32F, 1, 0, scale=1/32), cv2.Scharr(gray, cv2.CV_32F, 0, 1, scale=1/32)
    if method == "prewitt":
        kernel = np.float32([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) / 6
        return cv2.filter2D(gray, -1, kernel), cv2.filter2D(gray, -1, kernel.T)
    raise ValueError(f"Gradient operator must be sobel/prewitt/scharr: {method}")


def edge_response(img, method="canny"):
    gray = gray_float(img)
    if method in {"sobel", "prewitt", "scharr"}:
        gx, gy = gradients(gray, method)
        return cv2.magnitude(gx, gy)
    if method == "canny":
        return cv2.Canny(cv2.GaussianBlur(gray.astype(np.uint8), (5, 5), 1), 40, 120).astype(np.float32)
    if method == "laplacian":
        return np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    if method == "log":
        return np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (0, 0), 1.4), cv2.CV_32F, ksize=3))
    raise ValueError(f"Unknown edge operator: {method}")


def detect_edges(img, method="canny"):
    response = edge_response(img, method)
    if method == "canny":
        return response.astype(np.uint8)
    maximum = float(response.max())
    normalized = (response * 255 / max(maximum, 1e-6)).astype(np.uint8)
    return cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def sharpness(img):
    """Mean normalized Sobel magnitude; noise and contrast also affect this score."""
    return float(edge_response(img, "sobel").mean() / 255)


def gradient_orientation(img):
    """Dominant gradient normal in degrees modulo 180 (not guaranteed blur angle)."""
    gx, gy = gradients(img)
    magnitude = np.hypot(gx, gy)
    angles = np.mod(np.degrees(np.arctan2(gy, gx)), 180)
    histogram, _ = np.histogram(angles, bins=180, range=(0, 180), weights=magnitude)
    return float(np.argmax(histogram))


def orientation_estimates(img):
    """Sobel hints only: text layout can confound both skew and motion direction."""
    gx, gy = gradients(img)
    magnitude = np.hypot(gx, gy)
    angle = np.mod(np.degrees(np.arctan2(gy, gx)), 180)
    # Near-vertical normals of near-horizontal strokes give small text slopes.
    selected = (angle >= 75) & (angle <= 105) & (magnitude > 5)
    text_angle = float(np.average(angle[selected]-90, weights=magnitude[selected])) if selected.any() else 0.
    return {"textline_angle_sobel": text_angle,
            "motion_angle_gradient_heuristic": (gradient_orientation(img)+90) % 180}
