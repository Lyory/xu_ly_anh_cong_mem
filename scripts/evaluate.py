"""Evaluate paired synthetic images or unpaired real photographs without inventing GT."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import yaml
from src.pipeline import DocRestorer
from src.utils import image_files, read_rgb, identity_grid, write_rgb
from src.metrics import image_metrics, local_distortion, text_metrics
from src.classical.edge_detection import sharpness


def evaluate(input_dir, output, configurations, target_dir=None, bmap_dir=None,
             text_dir=None, ocr=False, lang="vie+eng", save_images=False):
    files = image_files(input_dir)
    if not files:
        raise ValueError(f"No images in {input_dir}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    names = [item["name"] for item in configurations]
    if len(names) != len(set(names)):
        raise ValueError("Configuration names must be unique")
    rows = []
    for item in configurations:
        options = dict(item.get("options", {}))
        if item.get("backend_config"):
            from src.models.pretrained_wrappers import load_backends
            options.update(load_backends(item["backend_config"]))
        modules = {"perspective": False, "deskew": False}
        modules.update(options.pop("modules", {}))
        restorer = None if item["name"] == "input" else DocRestorer(
            max_size=None, modules=modules, **options)
        if target_dir and restorer and (modules["perspective"] or modules["deskew"]):
            raise ValueError("Paired pixel metrics require fixed coordinates: disable perspective and deskew")
        for path in files:
            relative = path.relative_to(input_dir)
            image = read_rgb(path)
            started = time.perf_counter()
            result = restorer.restore(image) if restorer else {
                "image": image, "bmap": identity_grid(*image.shape[:2]), "metadata": {"models": {}}}
            restored = result["image"]
            row = {"image": str(relative), "configuration": item["name"],
                   "elapsed_seconds": time.perf_counter()-started,
                   "sharpness_sobel": sharpness(restored), "psnr": "", "ssim": "", "ms_ssim": "",
                   "ld_map_epe_px": "", "cer": "", "wer": "",
                   "smoke_checkpoint": any(x.get("smoke_only", False) for x in result["metadata"]["models"].values())}
            if target_dir:
                row.update(image_metrics(restored, read_rgb(Path(target_dir)/relative)))
            if bmap_dir:
                target_map = np.load(Path(bmap_dir)/relative.with_suffix(".npy"), allow_pickle=False)
                predicted_map = result["bmap"]
                if predicted_map is None and not restorer.modules["dewarp"]:
                    predicted_map = identity_grid(*image.shape[:2])
                if predicted_map is not None:
                    row["ld_map_epe_px"] = local_distortion(predicted_map, target_map, image.shape)
            if ocr:
                from src.ocr import recognize
                text = recognize(restored, lang)
                if text_dir:
                    reference = (Path(text_dir)/relative.with_suffix(".txt")).read_text(encoding="utf-8")
                    row.update(text_metrics(text, reference))
            if save_images:
                write_rgb(output.parent / "images" / item["name"] / relative, restored)
            rows.append(row)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    numeric = ["elapsed_seconds", "sharpness_sobel", "psnr", "ssim", "ms_ssim", "ld_map_epe_px", "cer", "wer"]
    for name in names:
        selected = [row for row in rows if row["configuration"] == name]
        group = {"configuration": name, "count": len(selected)}
        for metric in numeric:
            values = [float(row[metric]) for row in selected if row[metric] != ""]
            # Report perfect PSNRs separately rather than dropping infinity silently.
            group[metric] = float(np.mean(values)) if values else None
        group["perfect_psnr_count"] = sum(row["psnr"] == float("inf") for row in selected)
        summary.append(group)
    with output.with_name(output.stem+"_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    output.with_suffix(".json").write_text(json.dumps({
        "input_dir": str(input_dir), "target_dir": str(target_dir) if target_dir else None,
        "bmap_dir": str(bmap_dir) if bmap_dir else None,
        "text_dir": str(text_dir) if text_dir else None, "ocr": ocr, "lang": lang,
        "configs": configurations, "ld_definition": "normalized backward-map endpoint error in pixels, not DocUNet LD"
    }, ensure_ascii=False, indent=2))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--bmap", type=Path)
    parser.add_argument("--text", type=Path)
    parser.add_argument("--output", type=Path, default=Path("report/tables/evaluation.csv"))
    parser.add_argument("--configs", type=Path)
    parser.add_argument("--shadow-checkpoint")
    parser.add_argument("--dewarp-checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--lang", default="vie+eng")
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()
    if args.text and not args.ocr:
        parser.error("--text requires --ocr")
    configs = yaml.safe_load(args.configs.read_text()) if args.configs else [
        {"name": "input"}, {"name": "classical", "options": {"mode": "classical"}}]
    if not args.configs and (args.shadow_checkpoint or args.dewarp_checkpoint):
        for mode in ("dl", "hybrid"):
            configs.append({"name": mode, "options": {"mode": mode, "device": args.device,
                            "shadow_checkpoint": args.shadow_checkpoint, "dewarp_checkpoint": args.dewarp_checkpoint}})
    rows = evaluate(args.input, args.output, configs, args.target, args.bmap, args.text,
                    args.ocr, args.lang, args.save_images)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
