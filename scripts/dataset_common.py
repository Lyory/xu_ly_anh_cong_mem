import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from src.utils import image_files, read_rgb, write_rgb
from src.data.degradations import add_shadow, add_blur, add_noise_jpeg, add_warp


def generate(task, argv=None):
    parser = argparse.ArgumentParser(description=f"Generate paired {task} dataset")
    parser.add_argument("--clean", type=Path, default=Path("data/raw_clean"))
    parser.add_argument("--output", type=Path, default=Path(f"data/synthetic/{task}"))
    parser.add_argument("--variants", type=int, default=5)
    parser.add_argument("--size", type=int, default=512, help="Square training canvas; letterbox, no stretching")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.variants < 1 or args.size < 32:
        parser.error("variants >= 1 and size >= 32 required")
    if (args.output / "manifest.json").exists():
        parser.error("Dataset exists: choose a new output directory")
    if args.output.exists() and any(args.output.rglob("*.png")):
        # A new split must never silently retain obsolete pairs.
        parser.error("Output contains images; use a fresh output directory")
    files = image_files(args.clean)
    groups = {}
    for path in files:
        group = str(path.relative_to(args.clean).with_suffix("")).split("__p")[0]
        groups.setdefault(group, []).append(path)
    keys = sorted(groups)
    if len(keys) < 3:
        parser.error("Need >= 3 independent document groups for train/val/test; page suffix is __pNNNN")
    rng = np.random.default_rng(args.seed)
    rng.shuffle(keys)
    n_val = max(1, int(len(keys)*.1))
    n_test = max(1, int(len(keys)*.1))
    assignments = {key: ("test" if i < n_test else "val" if i < n_test+n_val else "train")
                   for i, key in enumerate(keys)}
    records = []
    for group in keys:
        split = assignments[group]
        for path in groups[group]:
            source = read_rgb(path)
            h, w = source.shape[:2]
            scale = args.size / max(h, w)
            resized = cv2.resize(source, (max(1, round(w*scale)), max(1, round(h*scale))), interpolation=cv2.INTER_AREA)
            clean = np.full((args.size, args.size, 3), 255, np.uint8)
            y, x = (args.size-resized.shape[0])//2, (args.size-resized.shape[1])//2
            clean[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
            identifier = hashlib.sha256(str(path.relative_to(args.clean)).encode()).hexdigest()[:12]
            for index in range(args.variants):
                name = f"{identifier}_{index:03d}"
                if task == "warp":
                    degraded, bmap = add_warp(clean, rng)
                    folder = args.output / split / "bmap"
                    folder.mkdir(parents=True, exist_ok=True)
                    np.save(folder / f"{name}.npy", bmap, allow_pickle=False)
                else:
                    degraded = add_noise_jpeg(add_blur(add_shadow(clean, rng), rng), rng,
                                              salt_pepper=.01 if rng.random() < .2 else 0.)
                write_rgb(args.output / split / "input" / f"{name}.png", degraded)
                write_rgb(args.output / split / "target" / f"{name}.png", clean)
                records.append({"id": name, "split": split, "source": str(path.relative_to(args.clean)), "group": group})
    metadata = {"task": task, "seed": args.seed, "size": args.size,
                "map_convention": "absolute normalized xy, align_corners=True, clean->warped",
                "samples": records}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print({split: sum(r["split"] == split for r in records) for split in ["train", "val", "test"]})
