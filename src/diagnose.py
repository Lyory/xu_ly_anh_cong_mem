import cv2
import numpy as np
from src.classical.shadow_removal import estimate_background
from src.classical.dewarp_classic import textline_curves
from src.classical.edge_detection import sharpness, gradient_orientation, orientation_estimates


def diagnose(img):
    """Heuristic scores, not calibrated probabilities. High blur_score = sharper."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bg = cv2.cvtColor(estimate_background(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    p10, p90 = np.percentile(bg, [10, 90])
    curves = textline_curves(img)
    curvature = float(np.median([np.ptp(offset) / img.shape[0] for _, offset in curves])) if curves else 0.
    return {"blur_score": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
            "shadow_score": float((p90-p10) / max(p90, 1)),
            "curvature_score": curvature,
            "sharpness_sobel": sharpness(img),
            "gradient_normal_degrees": gradient_orientation(img),
            **orientation_estimates(img),
            "textlines_detected": len(curves)}


def select_modules(scores, blur_threshold=110., shadow_threshold=.12, curvature_threshold=.006):
    return {"shadow": scores["shadow_score"] > shadow_threshold,
            "deblur": scores["blur_score"] < blur_threshold,
            "dewarp": scores["curvature_score"] > curvature_threshold}
