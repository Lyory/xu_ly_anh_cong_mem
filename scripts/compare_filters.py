import argparse
import csv
from pathlib import Path
import time
import cv2
import numpy as np
from src.utils import image_files, read_rgb, write_rgb
from src.data.degradations import add_shadow
from src.classical.edge_detection import METHODS as EDGES, detect_edges, sharpness
from src.classical.denoise import METHODS as FILTERS, denoise
from src.classical.detect_document import detect_document, order_corners
from src.metrics import image_metrics


def noise_variants(image, rng):
    gaussian = np.clip(image.astype(np.float32)+rng.normal(0, 15, image.shape), 0, 255).astype(np.uint8)
    mask = rng.random(image.shape[:2])
    salt = image.copy()
    salt[mask < .025] = 0
    salt[mask > .975] = 255
    _, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 25])
    jpeg = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return {"gaussian": gaussian, "salt_pepper": salt, "jpeg": jpeg}


def save_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Controlled edge and denoising comparisons")
    parser.add_argument("--clean", type=Path, default=Path("data/raw_clean"))
    parser.add_argument("--output", type=Path, default=Path("report/tables"))
    parser.add_argument("--figures", type=Path, default=Path("report/figures/filters"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    paths = image_files(args.clean)[:args.limit]
    if not paths:
        parser.error("No clean images")
    rng = np.random.default_rng(args.seed)
    filter_rows, edge_rows = [], []
    for index, path in enumerate(paths):
        clean = cv2.resize(read_rgb(path), (384, 512), interpolation=cv2.INTER_AREA)
        for noise, noisy in noise_variants(clean, rng).items():
            for name in ("input",)+FILTERS:
                tick = time.perf_counter()
                output = noisy if name == "input" else denoise(noisy, name)
                row = {"image": path.name, "noise": noise, "filter": name,
                       "elapsed_seconds": time.perf_counter()-tick, "sharpness_sobel": sharpness(output)}
                row.update(image_metrics(output, clean))
                filter_rows.append(row)
                if index == 0:
                    write_rgb(args.figures/f"{noise}_{name}.png", output)
        # Known page corners allow evaluating document boundary accuracy,
        # rather than treating greater edge magnitude as better detection.
        h, w = clean.shape[:2]
        corners = np.float32([[35, 28], [w-45, 50], [w-20, h-35], [20, h-25]])
        source = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
        matrix = cv2.getPerspectiveTransform(source, corners)
        page = cv2.warpPerspective(clean, matrix, (w, h), borderValue=(85, 70, 55))
        shadowed = add_shadow(page, rng)
        for condition, photo in [("clean", page), ("shadow", shadowed)]:
            for method in EDGES:
                tick = time.perf_counter()
                detected = detect_document(photo, edge_method=method)
                elapsed = time.perf_counter()-tick
                error = float(np.linalg.norm(order_corners(detected)-corners, axis=1).mean()) if detected is not None else ""
                edge_rows.append({"image": path.name, "condition": condition, "filter": method,
                                  "detected": detected is not None, "corner_error_px": error,
                                  "elapsed_seconds": elapsed})
                if index == 0:
                    edges = detect_edges(photo, method)
                    write_rgb(args.figures/f"edge_{condition}_{method}.png", cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))
    save_table(args.output/"denoise_comparison.csv", filter_rows)
    save_table(args.output/"edge_comparison.csv", edge_rows)
    print(f"Saved {len(filter_rows)} denoise rows and {len(edge_rows)} edge rows")


if __name__ == "__main__":
    main()
