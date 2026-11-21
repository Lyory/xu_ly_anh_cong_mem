"""Small, generated English text pages for smoke testing; not a research dataset."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from src.utils import write_rgb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw_clean"))
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()
    if args.count < 3:
        parser.error("Need >= 3 source documents")
    args.output.mkdir(parents=True, exist_ok=True)
    for index in range(args.count):
        image = np.full((768, 576, 3), 255, np.uint8)
        lines = [f"DOCUMENT {index+1:03d}", "IMAGE PROCESSING FINAL PROJECT", " "] + [
            f"Line {line:02d}: Restore readable document text." for line in range(1, 19)]
        for line, text in enumerate(lines):
            cv2.putText(image, text, (28, 45+line*30), cv2.FONT_HERSHEY_SIMPLEX,
                        .47 if line else .8, (20, 20, 20), 1, cv2.LINE_AA)
        path = args.output / f"demo_{index:03d}.png"
        if path.exists():
            raise FileExistsError(path)
        write_rgb(path, image)
        path.with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {args.count} smoke-test documents in {args.output}")


if __name__ == "__main__":
    main()
