"""Nắn ảnh theo đường bao tờ giấy, tinh chỉnh theo đường kẻ và hiệu chỉnh độ sáng."""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from src.classical.paper_segmentation import exclude_upper_attachment, segment_central_paper
from src.classical.form_dewarp import flatten_form, flatten_table, refine_lower_form


def _sample_edge_displacement(points, along_axis, length):
    """Lấy trung vị theo từng đoạn biên, trừ đường nối hai đầu để đo độ cong."""
    sample_count = 161
    bins = np.clip(np.rint(points[:, along_axis] * (sample_count - 1)), 0, sample_count - 1).astype(int)
    occupied = np.unique(bins)
    samples = [np.median(points[bins == index, 1-along_axis]) for index in occupied]
    values = np.interp(np.arange(sample_count), occupied, samples)
    values = gaussian_filter1d(median_filter(values, size=9), 3.2)
    values -= np.linspace(values[0], values[-1], sample_count)
    return np.interp(np.linspace(0, sample_count-1, length), np.arange(sample_count), values).astype(np.float32)


def _straighten_printed_layout(page):
    """Tìm khung biểu mẫu hoặc hàng bảng ở ảnh nhỏ rồi nắn ảnh gốc theo các đường kẻ."""
    scale = min(1., 960 / max(page.shape[:2]))
    analysis = cv2.resize(page, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(whiten_paper(analysis), cv2.COLOR_RGB2GRAY)
    restored, layout = flatten_form(page, gray)
    if layout:
        return restored, layout, "form"
    restored, layout = flatten_table(page, gray)
    return restored, layout, "table"


def _crop_page_for_uvdoc(image, mask):
    """Cắt vùng giấy và mặt nạ tương ứng, chừa lề cho UVDoc suy luận."""
    x, y, width, height = cv2.boundingRect(mask)
    scale_x, scale_y = image.shape[1]/mask.shape[1], image.shape[0]/mask.shape[0]
    crop = np.s_[round(y*scale_y):round((y+height)*scale_y),
                 round(x*scale_x):round((x+width)*scale_x)]
    region = image[crop]
    matte = cv2.resize(mask, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)[crop]
    margin = max(2, round(min(region.shape[:2]) * .03))
    region = cv2.copyMakeBorder(region, margin, margin, margin, margin,
                                cv2.BORDER_CONSTANT, value=(30, 30, 30))
    matte = cv2.copyMakeBorder(matte, margin, margin, margin, margin, cv2.BORDER_CONSTANT)
    return region, matte


def rectify_paper(image, dewarper=None):
    """Trả về trang chữ nhật và thông số nắn, hoặc cặp None khi đường bao thiếu tin cậy."""
    height, width = image.shape[:2]
    scale = min(1., 640 / max(height, width))
    small_width, small_height = round(width * scale), round(height * scale)
    if min(small_width, small_height) < 32:
        return None, None
    small = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_AREA)
    mask, upper_attachment = exclude_upper_attachment(segment_central_paper(small))
    coverage = cv2.countNonZero(mask) / mask.size
    if not .23 < coverage < .98 or not mask[small_height//2, small_width//2]:
        return None, None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    points = (contour if cv2.contourArea(contour, oriented=True) > 0 else contour[::-1]).reshape(-1, 2)

    directions = np.float32([[-1, 1, 1, -1], [-1, -1, 1, 1]])
    indices = np.argmax((points / [small_width, small_height]) @ directions, axis=0)
    corners = points[indices].astype(np.float32)
    _, tr, br, bl = (indices - indices[0]) % len(points)
    if not (0 < tr < br < bl and cv2.isContourConvex(corners)):
        return None, None
    if cv2.contourArea(corners) < .55 * cv2.contourArea(contour):
        return None, None

    unit_square = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    to_unit = cv2.getPerspectiveTransform(corners, unit_square)
    outline = cv2.perspectiveTransform(np.roll(points, -indices[0], axis=0)[None].astype(np.float32), to_unit)[0]
    scale_back = np.float32([(width-1)/(small_width-1), (height-1)/(small_height-1)])
    corners *= scale_back
    tl, top_right, bottom_right, bottom_left = corners
    out_width = round((np.linalg.norm(tl-top_right) + np.linalg.norm(bottom_left-bottom_right)) / 2)
    out_height = round((np.linalg.norm(tl-bottom_left) + np.linalg.norm(top_right-bottom_right)) / 2)
    if min(out_width, out_height) < 16:
        return None, None

    top = _sample_edge_displacement(outline[:tr+1], 0, out_width)
    right = _sample_edge_displacement(outline[tr:br+1], 1, out_height)
    bottom = _sample_edge_displacement(outline[br:bl+1], 0, out_width)
    left = _sample_edge_displacement(np.concatenate((outline[bl:], outline[:1])), 1, out_height)
    u, v = np.meshgrid(np.linspace(0, 1, out_width, dtype=np.float32),
                       np.linspace(0, 1, out_height, dtype=np.float32))
    # Nội suy độ lệch của hai mép đối diện theo tọa độ chuẩn hóa u, v.
    x = u + (1-u)*left[:, None] + u*right[:, None]
    y = v + (1-v)*top + v*bottom
    dx_du = np.diff(x, axis=1)[:-1]
    dy_dv = np.diff(y, axis=0)[:, :-1]
    dx_dv = np.diff(x, axis=0)[:, :-1]
    dy_du = np.diff(y, axis=1)[:-1]
    method = "paper_outline"
    if np.min(dx_du * dy_dv - dx_dv * dy_du) <= 0:
        # Định thức Jacobi không dương báo hiệu lưới gập, dùng phép nắn phối cảnh.
        x, y, method = u, v, "paper_mask_perspective"
    to_photo = cv2.getPerspectiveTransform(unit_square, corners)
    coords = cv2.perspectiveTransform(np.stack((x, y), axis=-1), to_photo)
    page = cv2.remap(image, coords[..., 0], coords[..., 1], cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    alpha = cv2.remap(mask, coords[..., 0] / scale_back[0], coords[..., 1] / scale_back[1],
                      cv2.INTER_NEAREST)
    page[alpha == 0] = 255
    margin = max(2, round(min(out_width, out_height) * .006))
    page = cv2.copyMakeBorder(page, margin, margin, margin, margin,
                              cv2.BORDER_CONSTANT, value=(255, 255, 255))
    geometry = {"method": method, "corners": corners.tolist(),
                "upper_attachment_removed": upper_attachment}
    page, layout, kind = _straighten_printed_layout(page)
    if dewarper is not None and (layout or upper_attachment):
        # Chỉ nhận dự đoán khi còn phát hiện được các đường kẻ của trang.
        region, matte = _crop_page_for_uvdoc(image, mask)
        predicted, grid = dewarper(region)
        x_map = (grid[..., 0] + 1) * ((region.shape[1]-1)/2)
        y_map = (grid[..., 1] + 1) * ((region.shape[0]-1)/2)
        warped_matte = cv2.remap(matte, x_map, y_map, cv2.INTER_NEAREST)
        distance = cv2.distanceTransform(warped_matte, cv2.DIST_L2, 5)
        feather = max(2, round(min(region.shape[:2]) * .007))
        alpha = np.minimum(distance / feather, 1)[..., None]
        predicted = np.rint(255 - (255 - predicted)*alpha).astype(np.uint8)
        refined, refined_layout, refined_kind = _straighten_printed_layout(predicted)
        if refined_layout and (kind != "form" or refined_kind == "form"):
            if refined_kind == "form":
                _, second_grid = dewarper(predicted)
                adjusted = refine_lower_form(predicted, second_grid, refined_layout["frame_rows"])
                if adjusted is not None:
                    candidate, candidate_layout, candidate_kind = _straighten_printed_layout(adjusted)
                    if candidate_kind == "form" and candidate_layout["guides"] == refined_layout["guides"]:
                        ratios = np.array(candidate.shape[:2])/refined.shape[:2]
                        if np.max(np.abs(ratios-1)) < .05:
                            refined, refined_layout = candidate, candidate_layout
                            refined_layout["local_refinement"] = True
            # Khôi phục tỉ lệ từ bốn mép giấy sau khi UVDoc nắn trong hình chữ nhật bao.
            model_aspect = predicted.shape[1] / predicted.shape[0]
            width_ratio = (out_width / out_height) / model_aspect
            width = round(refined.shape[1] * width_ratio)
            page = cv2.resize(refined, (width, refined.shape[0]), interpolation=cv2.INTER_CUBIC)
            layout, kind = refined_layout, refined_kind
            geometry["form_dewarper"] = "uvdoc"
    if layout:
        geometry[f"{kind}_rectification"] = layout
    return page, geometry


def whiten_paper(image, table_lower_edge=None):
    """Chia từng kênh màu cho nền sáng ước lượng để làm trắng giấy và giữ màu mực."""
    height, width = image.shape[:2]
    scale = min(1., (960 if table_lower_edge is not None else 640) / max(height, width))
    small = cv2.resize(image, (max(1, round(width*scale)), max(1, round(height*scale))),
                       interpolation=cv2.INTER_AREA)
    size = max(5, round(min(small.shape[:2]) * .026) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    background = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), size * .22)
    if table_lower_edge is not None:
        # Dùng cửa sổ nhỏ dưới bảng để bám nếp gấp, giữ cửa sổ lớn ở vùng nhiều chữ.
        detail_size = max(5, round(min(small.shape[:2]) * .010) | 1)
        detail_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (detail_size, detail_size))
        detail = cv2.morphologyEx(small, cv2.MORPH_CLOSE, detail_kernel)
        detail = cv2.GaussianBlur(detail, (0, 0), detail_size * .19)
        rows = np.linspace(0, 1, small.shape[0], dtype=np.float32)
        blend = np.clip((rows - table_lower_edge - .10) / .05, 0, 1)[:, None, None]
        background = (1-blend)*background + blend*detail
    background = cv2.resize(background, (width, height)).astype(np.float32)
    ratio = image.astype(np.float32) / np.maximum(background, 40)
    return np.rint(np.clip(1 - (.963 - ratio) * 1.45, 0, 1) * 255).astype(np.uint8)
