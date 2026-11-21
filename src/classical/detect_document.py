import cv2
import numpy as np
from src.classical.edge_detection import detect_edges


def order_corners(points):
    points = np.asarray(points, np.float32).reshape(4, 2)
    center = points.mean(0)
    points = points[np.argsort(np.arctan2(points[:, 1]-center[1], points[:, 0]-center[0]))]
    return np.roll(points, -np.argmin(points.sum(1)), axis=0)


def detect_document(img, min_area=.20, edge_method="canny"):
    """Return TL, TR, BR, BL corners or None when a confident quad is absent."""
    h, w = img.shape[:2]
    scale = min(1., 1000 / max(h, w))
    small = cv2.resize(img, None, fx=scale, fy=scale)
    edges = detect_edges(small, edge_method)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) < min_area * small.shape[0] * small.shape[1]:
            continue
        quad = cv2.approxPolyDP(contour, .02 * cv2.arcLength(contour, True), True)
        if len(quad) == 4 and cv2.isContourConvex(quad):
            return order_corners(quad.reshape(4, 2) / scale)
    return None


def rectify_document(img, corners=None):
    corners = detect_document(img) if corners is None else order_corners(corners)
    if corners is None:
        return img.copy(), None
    tl, tr, br, bl = corners
    width = max(16, int(max(np.linalg.norm(tr-tl), np.linalg.norm(br-bl))))
    height = max(16, int(max(np.linalg.norm(bl-tl), np.linalg.norm(br-tr))))
    target = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target)
    return cv2.warpPerspective(img, matrix, (width, height), borderValue=(255, 255, 255)), matrix
