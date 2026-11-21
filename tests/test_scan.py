import cv2
import numpy as np

from src.classical.binarize import remove_edge_marks, render_scan_style
from src.classical.form_dewarp import flatten_form, flatten_table, refine_lower_form
from src.classical.paper_segmentation import exclude_upper_attachment
from src.pipeline import DocRestorer


def test_scan_crops_curved_paper_and_preserves_blue_marks():
    image = np.full((480, 400, 3), 38, np.uint8)
    outline = np.int32([[65, 55], [192, 36], [335, 58], [355, 227], [367, 415],
                        [211, 424], [41, 416], [51, 235]])
    mask = cv2.fillPoly(np.zeros(image.shape[:2], np.uint8), [outline], 1).astype(bool)
    shade = np.linspace(220, 132, image.shape[0]).astype(np.uint8)
    image[mask] = np.broadcast_to(shade[:, None, None], image.shape)[mask]
    for x, y in ((88, 103), (307, 103), (78, 376), (327, 376)):
        cv2.line(image, (x-10, y), (x+10, y), (32, 56, 125), 3, cv2.LINE_AA)

    result = DocRestorer(mode="scan").restore(image)
    restored = result["image"]
    height, width = restored.shape[:2]
    assert result["metadata"]["paper_geometry"]["method"] == "paper_outline"
    assert height < image.shape[0] and width < image.shape[1]
    assert np.median(restored[height//2:, width//4:3*width//4]) >= 245
    blue_ink = restored[..., 2].astype(int) - restored[..., 0] > 45
    for rows in (slice(None, height//2), slice(height//2, None)):
        for cols in (slice(None, width//2), slice(width//2, None)):
            assert blue_ink[rows, cols].sum() > 20


def test_scan_keeps_framing_when_page_fills_image():
    image = np.full((200, 180, 3), 218, np.uint8)
    cv2.line(image, (30, 90), (150, 90), (20, 45, 100), 2)
    result = DocRestorer(mode="scan").restore(image)
    assert result["image"].shape == image.shape
    assert "paper_geometry" not in result["metadata"]
    assert result["metadata"]["warnings"]


def test_scan_excludes_upper_sheet_attached_to_central_page():
    image = np.full((640, 480, 3), 34, np.uint8)
    cv2.rectangle(image, (56, 138), (423, 586), (196, 196, 196), -1)
    cv2.rectangle(image, (275, 13), (432, 145), (205, 205, 205), -1)
    cv2.line(image, (292, 82), (406, 82), (25, 45, 140), 4)
    cv2.line(image, (103, 228), (270, 228), (142, 42, 26), 4)

    result = DocRestorer(mode="scan").restore(image)
    assert result["metadata"]["paper_geometry"]["upper_attachment_removed"]
    restored = result["image"].astype(np.int16)
    assert restored.shape[0] < 510
    assert not np.any(restored[..., 2] - restored[..., 0] > 40)
    assert np.count_nonzero(restored[..., 0] - restored[..., 2] > 40) > 100


def test_form_guides_straighten_curved_rules():
    height, width = 480, 400
    gray = np.full((height, width), 255, np.uint8)
    cols = np.arange(43, 357)
    curve = (16*np.sin(np.pi*(cols-cols[0])/(cols[-1]-cols[0]))).astype(np.int32)
    for row in (58, 125, 359, 418):
        line = np.column_stack((cols, row + curve)).astype(np.int32)
        cv2.polylines(gray, [line], False, 18, 2)
    for col in (cols[0], cols[-1]):
        cv2.line(gray, (col, 125), (col, 359), 18, 2)
    page = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    restored, details = flatten_form(page, gray)
    assert details["guides"] == 6 and restored.shape[0] < height
    rule = restored[:45, 55:-55, 0] < 120
    assert rule.any(axis=0).all()
    assert np.ptp(rule.argmax(axis=0)) <= 3


def test_scan_trims_sloping_top_attachment_without_clipping_left_header():
    mask = np.zeros((640, 480), np.uint8)
    invoice = np.int32([[21, 153], [71, 87], [373, 53], [374, 385],
                        [412, 627], [22, 611]])
    cv2.fillConvexPoly(mask, invoice, 255)
    cv2.rectangle(mask, (304, 0), (475, 138), 255, -1)
    trimmed, detected = exclude_upper_attachment(mask)
    assert detected
    assert trimmed[87, 154] == 255
    assert trimmed[74, 365] == 255
    assert trimmed[40, 405] == 0 and trimmed[132, 455] == 0


def test_invoice_table_straightens_internal_rows_and_leaves_stamp_in_place():
    height, width = 480, 400
    gray = np.full((height, width), 255, np.uint8)
    cols = np.arange(40, 365)
    curve = np.rint(10*np.sin(np.linspace(0, np.pi, cols.size))).astype(int)
    for row in (80, 102, 124, 146, 168):
        points = np.column_stack((cols, row+curve)).astype(np.int32)
        cv2.polylines(gray, [points], False, 20, 2)
    for col in (cols[0], cols[-1], 135, 260):
        offset = curve[col-cols[0]]
        cv2.line(gray, (col, 80+offset), (col, 168+offset), 20, 2)
    page = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(page, (160, 288), (240, 314), (160, 22, 26), 2)

    restored, details = flatten_table(page, gray)
    assert details["guides"] == 5 and details["max_row_bend_px"] >= 8
    assert restored.shape == page.shape
    rule = restored[75:96, 65:-65, 0] < 120
    assert rule.any(axis=0).all()
    assert np.ptp(rule.argmax(axis=0)) <= 2
    np.testing.assert_array_equal(restored[275:330], page[275:330])


def test_edge_cleanup_preserves_rules_and_small_interior_marks():
    page = np.full((500, 400, 3), 255, np.uint8)
    cv2.line(page, (30, 6), (370, 6), (8, 8, 8), 2)
    cv2.line(page, (3, 205), (10, 223), (7, 7, 7), 2)
    cv2.circle(page, (70, 215), 3, (7, 7, 7), -1)
    cleaned = remove_edge_marks(page)
    assert np.all(cleaned[197:228, :20] == 255)
    assert cleaned[6, 190, 0] == 8
    assert cleaned[215, 70, 0] == 7


def test_scan_styles_keep_color_or_gray_or_binary_as_requested():
    page = np.full((160, 130, 3), 255, np.uint8)
    cv2.line(page, (20, 60), (110, 60), (37, 75, 156), 1, cv2.LINE_AA)
    color = render_scan_style(page, "magic-color")
    gray = render_scan_style(page, "soft-bw")
    binary = render_scan_style(page, "sauvola")
    assert color[60, 60, 2] > color[60, 60, 0]
    assert np.all(gray[..., 0] == gray[..., 2])
    assert 0 < gray[60, 60, 0] < 255
    assert set(np.unique(binary)) == {0, 255}


def test_local_form_refinement_discards_global_rescale_and_translation():
    rng = np.random.default_rng(12)
    page = rng.integers(0, 256, (150, 130, 3), dtype=np.uint8)
    x, y = np.meshgrid(np.linspace(-1, 1, page.shape[1], dtype=np.float32),
                       np.linspace(-1, 1, page.shape[0], dtype=np.float32))
    prediction = np.stack((.91*x + .04, .94*y - .03), axis=-1)
    refined = refine_lower_form(page, prediction, (.16, .88))
    np.testing.assert_array_equal(refined, page)
