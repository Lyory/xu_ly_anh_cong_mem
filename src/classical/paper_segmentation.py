"""Tách tờ giấy chính khỏi nền và các vật thể xung quanh."""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter


def _fill_central_contour(mask):
    """Tô kín đường bao chứa tâm ảnh, hoặc vùng lớn nhất nếu tâm nằm ngoài."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(mask)
    if contours:
        height, width = mask.shape
        centered = [contour for contour in contours
                    if cv2.pointPolygonTest(contour, (width/2, height/2), False) >= 0]
        cv2.drawContours(result, [max(centered or contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return result


def exclude_upper_attachment(mask):
    """Dựa vào bề rộng từng hàng để tách mảnh giấy dính phía trên trang chính."""
    height = mask.shape[0]
    widths = median_filter(np.count_nonzero(mask, axis=1), size=7)
    body_width = np.median(widths[height*2//5:height*7//10])
    wide_threshold = .8 * body_width
    wide_rows = np.flatnonzero(widths[:height//2] > wide_threshold)
    if not wide_rows.size:
        return mask, False
    top = wide_rows[0]
    upper_rows = np.flatnonzero(widths[:top] > .13 * body_width)
    if not upper_rows.size or top-upper_rows[0] < height * .075:
        return mask, False
    upper_width = np.median(widths[upper_rows[0]:top])
    if not .13 * body_width < upper_width < .66 * body_width:
        return mask, False
    margin = max(6, round(height * .024))
    if (np.median(widths[top-margin:top]) >= body_width * .72
            or np.min(widths[top+margin:top+3*margin]) < wide_threshold):
        return mask, False
    # Với mép giấy xiên, cắt ngang có thể làm mất tiêu đề.
    narrow_mask = mask[upper_rows[0]:top-margin]
    if cv2.countNonZero(narrow_mask):
        _, cols = np.nonzero(narrow_mask)
        attachment_right = np.median(cols) > mask.shape[1] / 2
        oriented = mask if attachment_right else np.ascontiguousarray(mask[:, ::-1])
        left = (oriented > 0).argmax(axis=1)
        body_left = np.median(left[height*2//5:height*7//10])
        if left[top] - body_left > .13 * body_width:
            separated = _exclude_side_attachment(oriented)
            if separated is not None:
                return separated if attachment_right else separated[:, ::-1].copy(), True
    trimmed = mask.copy()
    trimmed[:top+margin] = 0
    return _fill_central_contour(trimmed), True


def _predict_edge_positions(x, y, target):
    """Khớp đường thẳng bằng mất mát Huber rồi ngoại suy vị trí mép giấy."""
    dx, dy, cx, cy = cv2.fitLine(np.float32(np.column_stack((x, y))), cv2.DIST_HUBER,
                                0, .01, .01).ravel()
    return (dy/dx) * (target-cx) + cy


def _exclude_side_attachment(mask):
    """Dùng mép trên và mép phải của thân trang để loại phần giấy dính bên cạnh."""
    height, width = mask.shape
    body = mask[round(.38*height):round(.65*height)] > 0
    left = np.median(body.argmax(axis=1))
    right = np.median(width-1-body[:, ::-1].argmax(axis=1))
    x, y = np.arange(width), np.arange(height)
    active = np.flatnonzero(mask.any(axis=0) & (x >= left+.24*(right-left))
                            & (x <= left+.71*(right-left)))
    top = (mask > 0).argmax(axis=0)
    reliable = active[np.abs(top[active] - np.median(top[active])) < .028*height] if active.size else active
    if reliable.size < 12:
        return None
    cap = _predict_edge_positions(reliable, top[reliable], x) - .006*height
    rows = y[round(.375*height):round(.6*height)]
    side = width-1-(mask[rows, ::-1] > 0).argmax(axis=1)
    edge = np.clip(_predict_edge_positions(rows, side, y) + .004*width, right-.06*width, right+.05*width)
    keep = (y[:, None] >= cap) & ((y[:, None] >= .38*height) | (x <= edge[:, None]))
    return _fill_central_contour(np.where(keep, mask, 0).astype(np.uint8))


def segment_central_paper(image):
    """Ước lượng vùng giấy sáng ở giữa rồi tinh chỉnh mặt nạ bằng GrabCut."""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    middle = blurred[:, width//4:3*width//4]
    baseline = max(45., np.percentile(middle[height//4:3*height//4], 20) * .77)
    row_light = gaussian_filter1d(np.percentile(middle, 25, axis=1), 14)
    threshold = np.clip(row_light * .72, baseline * .89, baseline)
    # Ở sát biên ảnh, dùng ngưỡng chung vì các cột giữa có thể chứa nền bàn.
    threshold[:max(1, round(height * .18))] = baseline
    threshold[-max(1, round(height * .025)):] = baseline
    threshold = gaussian_filter1d(threshold, 6)[:, None]

    gaps = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bright = np.where(blurred > threshold, 255, 0).astype(np.uint8)
    initial = _fill_central_contour(cv2.morphologyEx(bright, cv2.MORPH_CLOSE, gaps))
    inner_distance = cv2.distanceTransform(initial, cv2.DIST_L2, 5)
    outer_distance = cv2.distanceTransform(255 - initial, cv2.DIST_L2, 5)
    paper = (inner_distance > 18) & (blurred > threshold - 5)
    background = (outer_distance > 40) | ((initial == 0) & (blurred < threshold - 9))
    if not (paper.any() and background.any()):
        return initial

    seeds = np.where(initial > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    seeds[paper] = cv2.GC_FGD
    seeds[background] = cv2.GC_BGD
    cv2.grabCut(cv2.GaussianBlur(image, (0, 0), 1.8), seeds, None,
                np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
    # Nối khe hẹp do bóng hoặc nếp gấp kéo đến mép giấy.
    creases = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    return _fill_central_contour(cv2.morphologyEx((seeds % 2) * 255, cv2.MORPH_CLOSE, creases))
