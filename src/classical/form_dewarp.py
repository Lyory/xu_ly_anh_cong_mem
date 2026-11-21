"""Nắn biểu mẫu và bảng dựa trên đường kẻ in trên giấy."""

from collections import Counter
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter


def _trace_frame_edges(frame, across_columns):
    """Dò hai cạnh đối diện của khung qua các hàng hoặc cột có đủ nét mực."""
    if not across_columns:
        frame = frame.T
    occupied = frame.any(axis=0)
    near = frame.argmax(axis=0)
    far = frame.shape[0] - 1 - frame[::-1].argmax(axis=0)
    # Loại các hàng hoặc cột chỉ nhìn thấy một cạnh, thường gặp gần góc bo.
    supported = np.flatnonzero(occupied & (far - near > .55 * np.max(far[occupied] - near[occupied])))
    if across_columns and supported.size:
        limit = round(.09 * (supported[-1] - supported[0]))
        supported = supported[limit:len(supported)-limit] if limit else supported
    if supported.size < 12:
        return []
    guides = []
    for edge in (near, far):
        center = np.median(edge[supported])
        # Nét ngang ở góc có thể bị nhận nhầm thành cạnh dọc.
        reliable = supported[np.abs(edge[supported] - center) < .10*frame.shape[0]]
        if reliable.size < 12:
            return []
        guides.append(_smooth_guide(edge, reliable))
    return guides


def _smooth_guide(edge, supported):
    """Nội suy các điểm thiếu rồi lọc trung vị và Gaussian để làm trơn đường dẫn."""
    curve = np.interp(np.arange(len(edge)), supported, edge[supported])
    return gaussian_filter1d(median_filter(curve, size=9), max(2, len(edge) * .012))


def _trace_rule_center(component):
    """Tính đường tâm của nét kẻ từ trung điểm của hai biên và trả về khoảng cột có mực."""
    supported = np.flatnonzero(component.any(axis=0))
    near = component.argmax(axis=0)
    far = component.shape[0] - 1 - component[::-1].argmax(axis=0)
    return _smooth_guide((near + far) / 2, supported), (supported[0], supported[-1])


def _interpolate_sampling_axis(guides, length, min_spacing=.015):
    """Nội suy tọa độ lấy mẫu giữa các đường dẫn theo từng cột, loại các đường giao nhau."""
    guides = np.asarray(guides)
    anchors = np.concatenate(([0], np.median(guides, axis=1), [length-1]))
    boundaries = np.vstack((np.zeros(guides.shape[1]), guides,
                            np.full(guides.shape[1], length-1)))
    if np.min(np.diff(boundaries, axis=0)) < length * min_spacing or np.min(np.diff(anchors)) < length * min_spacing:
        return None
    return np.stack([np.interp(np.arange(length), anchors, column) for column in boundaries.T],
                    axis=1).astype(np.float32)


def _ink_components(gray):
    """Gán nhãn các vùng mực liên thông sau khi loại dải nhiễu sát mép."""
    height, width = gray.shape
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 8)
    mx, my = max(1, round(width * .035)), max(1, round(height * .022))
    ink[:, :mx] = ink[:, -mx:] = ink[:my] = ink[-my:] = 0
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    return labels, stats


def _encloses_frame(component, width, height):
    """Kiểm tra diện tích đường bao có đủ lớn so với hình chữ nhật ngoại tiếp."""
    contours, _ = cv2.findContours(np.uint8(component), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(map(cv2.contourArea, contours)) > .65 * width * height


def flatten_form(page, gray):
    """Nắn hai cạnh bên và bốn đường ngang, chừa lề cho tiêu đề và ghi chú."""
    height, width = gray.shape
    labels, stats = _ink_components(gray)
    frames = []
    for index, (_, _, w, h, _) in enumerate(stats[1:], 1):
        if w < .46 * width or h < .30 * height:
            continue
        if _encloses_frame(labels == index, w, h):
            frames.append((w*h, index))
    if not frames:
        return page, None
    frame_index = max(frames)[1]
    frame = labels == frame_index
    sides = _trace_frame_edges(frame, across_columns=False)
    horizontal = _trace_frame_edges(frame, across_columns=True)
    if not (sides and horizontal):
        return page, None

    # Hai đường kẻ ngoài khung giúp nhận biết toàn bộ biểu mẫu.
    top, bottom = horizontal
    upper, lower = [], []
    for index, (_, y, w, h, _) in enumerate(stats[1:], 1):
        if w < .55 * width or h > .105 * height:
            continue
        if y+h < np.median(top) - .02*height:
            upper.append((w, index))
        elif y > np.median(bottom) + .012*height:
            lower.append((w, index))
    if not (upper and lower):
        return page, None
    upper_guide, upper_span = _trace_rule_center(labels == max(upper)[1])
    lower_guide, lower_span = _trace_rule_center(labels == max(lower)[1])

    x_map = _interpolate_sampling_axis(sides, width)
    if x_map is None:
        return page, None
    x_map = x_map.T.copy()
    cols = np.arange(width, dtype=np.float32)
    guide_rows = [upper_guide, top, bottom, lower_guide]
    guide_rows = [np.interp(x_map[round(np.median(guide))], cols, guide) for guide in guide_rows]
    y_map = _interpolate_sampling_axis(guide_rows, height)
    if y_map is None:
        return page, None
    x_map = cv2.remap(x_map, np.broadcast_to(cols, (height, width)).copy(), y_map, cv2.INTER_LINEAR)

    # Chừa lề cho tiêu đề và ghi chú ngoài khung.
    left = max(0, int(min(np.median(sides[0]), upper_span[0], lower_span[0]) - .09*width))
    right = min(width, int(max(np.median(sides[1]), upper_span[1], lower_span[1]) + .07*width))
    start = max(0, int(np.median(guide_rows[0]) - .043*height))
    end = min(height, int(np.median(guide_rows[-1]) + .037*height))
    if right - left < .6*width or end - start < .65*height:
        return page, None

    full_height, full_width = page.shape[:2]
    scale_x, scale_y = (full_width-1)/(width-1), (full_height-1)/(height-1)
    roi = np.s_[start:end, left:right]
    size = (round((right-left)*scale_x), round((end-start)*scale_y))
    map_x = cv2.resize(x_map[roi], size) * scale_x
    map_y = cv2.resize(y_map[roi], size) * scale_y
    restored = cv2.remap(page, map_x, map_y, cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return restored, {"guides": len(sides) + len(guide_rows),
                      "crop": [left, start, right, end],
                      "frame_rows": [float(np.median(guide_rows[1])/height),
                                     float(np.median(guide_rows[2])/height)],
                      "analysis_size": [width, height]}


def refine_lower_form(page, backward_map, frame_rows):
    """Tách dịch chuyển phi affine của UVDoc để chỉnh cục bộ nửa dưới biểu mẫu."""
    field = cv2.resize(backward_map, (128, 128), interpolation=cv2.INTER_LINEAR)
    if not np.isfinite(field).all():
        return None
    axis = np.linspace(-1, 1, 128, dtype=np.float32)
    x, y = np.meshgrid(axis, axis)
    core = (np.abs(x) < .75) & (np.abs(y) < .75)
    samples = np.column_stack((x[core], y[core], np.ones(np.count_nonzero(core))))
    affine = np.linalg.lstsq(samples, field[core], rcond=None)[0].T
    if not .4 < np.linalg.det(affine[:, :2]) < 2:
        return None
    inverse = cv2.invertAffineTransform(affine)
    u = (inverse[0, 0]*x + inverse[0, 1]*y + inverse[0, 2] + 1) * 63.5
    v = (inverse[1, 0]*x + inverse[1, 1]*y + inverse[1, 2] + 1) * 63.5
    fitted = x[..., None]*affine[:, 0] + y[..., None]*affine[:, 1] + affine[:, 2]
    residual = (field-fitted).astype(np.float32)
    registered = cv2.remap(residual, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
    top, bottom = frame_rows
    position = ((y+1)/2 - top)/(bottom-top)
    weight = (.8 * np.clip((position-.31)/.21, 0, 1) *
              np.clip((1.08-position)/.23, 0, 1) * np.clip((.95-np.abs(x))/.20, 0, 1))
    dx = weight * registered[..., 0]
    dy = weight * registered[..., 1]
    if max(np.max(np.abs(dx)), np.max(np.abs(dy))) > .075:
        return None
    dx_du = np.diff(x+dx, axis=1)[:-1]
    dy_dv = np.diff(y+dy, axis=0)[:, :-1]
    dx_dv = np.diff(x+dx, axis=0)[:, :-1]
    dy_du = np.diff(y+dy, axis=1)[:-1]
    if np.min(dx_du*dy_dv - dx_dv*dy_du) < .22 * (2/127)**2:
        return None
    height, width = page.shape[:2]
    px, py = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    px += cv2.resize(dx, (width, height)) * ((width-1)/2)
    py += cv2.resize(dy, (width, height)) * ((height-1)/2)
    return cv2.remap(page, px, py, cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def _trace_table_rows(frame, x, width, height):
    """Lần theo tâm các hàng bảng ở những cột có cùng số đường kẻ."""
    columns = {}
    for col in range(x, x+width):
        rows = np.flatnonzero(frame[:, col])
        if rows.size:
            strokes = np.split(rows, np.flatnonzero(np.diff(rows) > 1)+1)
            centers = [np.mean(stroke) for stroke in strokes if len(stroke) <= max(3, height*.06)]
            if len(centers) >= 4 and np.min(np.diff(centers)) > height*.05:
                columns[col] = centers
    if not columns:
        return []
    count, frequency = Counter(map(len, columns.values())).most_common(1)[0]
    if frequency < .50 * width:
        return []
    span = np.array([col for col, centers in columns.items() if len(centers) == count])
    if span[-1] - span[0] < .82 * width:
        return []
    samples = np.array([columns[col] for col in span]).T
    interpolated = [np.interp(np.arange(frame.shape[1]), span, row) for row in samples]
    # Dùng cùng độ rộng lọc để giữ thứ tự các đường dẫn song song.
    return [gaussian_filter1d(median_filter(row, size=9), max(2, width*.009))
            for row in interpolated]


def flatten_table(page, gray):
    """Làm thẳng hàng bảng trên hóa đơn và giảm dần biến dạng ra ngoài bảng."""
    height, width = gray.shape
    labels, stats = _ink_components(gray)
    for index in np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1]+1:
        x, _, w, h, _ = stats[index]
        if not (w > .46*width and .065*height < h < .33*height):
            continue
        frame = labels == index
        if not _encloses_frame(frame, w, h):
            continue
        guides = _trace_table_rows(frame, x, w, h)
        if not guides:
            continue
        # Giảm dần độ nắn về 0 ngoài bảng để bảo toàn tiêu đề, chân trang và con dấu.
        margin = height * .055
        anchor_top = max(1, np.floor(guides[0].min() - margin))
        anchor_bottom = min(height-2, np.ceil(guides[-1].max() + margin))
        anchors = [np.full(width, anchor_top), *guides, np.full(width, anchor_bottom)]
        y_map = _interpolate_sampling_axis(anchors, height, min_spacing=.004)
        if y_map is None:
            continue
        rows = np.arange(height, dtype=np.float32)[:, None]
        correction = y_map - rows
        full_height, full_width = page.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(full_width, dtype=np.float32),
                                     np.arange(full_height, dtype=np.float32))
        delta = cv2.resize(correction, (full_width, full_height)) * ((full_height-1)/(height-1))
        restored = cv2.remap(page, grid_x, grid_y + delta, cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return restored, {"guides": len(guides), "table_lower_edge": float(np.median(guides[-1])/height),
                          "max_row_bend_px": round(float(max(np.ptp(row) for row in guides)), 2),
                          "analysis_size": [width, height]}
    return page, None
