import cv2
import numpy as np

SCAN_STYLES = ("magic-color", "soft-bw", "sauvola")


def clahe(img, clip_limit=2.):
    """Tăng tương phản cục bộ trên kênh sáng L của ảnh Lab."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clip_limit, (8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _window_mean(values, window):
    """Tính trung bình cửa sổ vuông bằng hiệu hai tổng tích lũy trên từng trục."""
    radius = window // 2
    for axis in (0, 1):
        # Với tổng tiền tố S, tổng của w mẫu từ i là S[i+w] - S[i].
        rows = np.swapaxes(values, 0, axis)
        padded = np.pad(rows, ((radius, radius), (0, 0)), mode="symmetric")
        prefix = np.empty((len(padded) + 1, rows.shape[1]), dtype=np.float64)
        prefix[0] = 0
        np.cumsum(padded, axis=0, dtype=np.float64, out=prefix[1:])
        values = np.swapaxes(prefix[window:] - prefix[:-window], 0, axis)
    return (values / (window * window)).astype(np.float32)


def binarize(img, window=31, k=.2, max_threshold=None):
    """Tách mực bằng ngưỡng Sauvola T = m × [1 + k × (s/128 − 1)]."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    window = min(window | 1, (min(gray.shape) - 1) | 1)
    mean = _window_mean(gray, window)
    variance = _window_mean(gray * gray, window) - mean**2
    threshold = mean * (1 + k * (np.sqrt(np.maximum(variance, 0)) / 128 - 1))
    if max_threshold is not None:
        threshold = np.minimum(threshold, max_threshold)
    return cv2.cvtColor(np.where(gray > threshold, 255, 0).astype(np.uint8), cv2.COLOR_GRAY2RGB)


def remove_edge_marks(image):
    """Xóa cụm mực nhỏ sát mép, giữ các đường kẻ dài và nội dung bên trong."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(gray < 210))
    x, y, w, h, area = stats.T
    edge = ((x < .026*width) | (x+w > .974*width) |
            (y < .020*height) | (y+h > .980*height))
    compact = (w < .15*width) & (h < .15*height) & (area < .001*width*height)
    remove = edge & compact
    remove[0] = False
    output = image.copy()
    output[remove[labels]] = 255
    return output


def render_scan_style(image, style, clean_edges=False):
    """Xuất ảnh màu, ảnh xám theo đường cong gamma hoặc ảnh nhị phân Sauvola."""
    if style == "soft-bw":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        tone = np.clip((gray-24)/224, 0, 1)**1.15
        image = cv2.cvtColor(np.rint(tone*255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    elif style == "sauvola":
        window = max(35, round(.04*min(image.shape[:2])) | 1)
        image = binarize(image, window, k=.16, max_threshold=222)
    elif style != "magic-color":
        raise ValueError(f"Unknown scan style: {style}")
    return remove_edge_marks(image) if clean_edges else image
