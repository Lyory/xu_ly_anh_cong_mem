"""Measure OCR CER on paired, manually annotated regions of printed text."""

import argparse
import csv
import json
from pathlib import Path
import unicodedata

from jiwer import process_characters
from src.classical.preprocess import load_image
from src.ocr import recognize


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.translate(str.maketrans({"–": "-", "—": "-", "−": "-"}))
    return " ".join(text.split())


def crop(image, box):
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(f"Invalid normalized crop: {box}")
    height, width = image.shape[:2]
    return image[round(y0*height):round(y1*height), round(x0*width):round(x1*width)].copy()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--restored", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotations = json.loads(args.regions.read_text(encoding="utf-8"))
    if not annotations["regions"]:
        parser.error("At least one printed region is required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    variants = [("input", Path(annotations["input"]), "input_box")]
    variants.extend((path.stem, path, "restored_box") for path in args.restored)
    rows, summary = [], []
    for variant, path, box_key in variants:
        image = load_image(path)
        edits = characters = 0
        for region in annotations["regions"]:
            reference = normalize(region["reference"])
            prediction = normalize(recognize(crop(image, region[box_key]),
                                             annotations["lang"], psm=annotations["psm"]))
            result = process_characters(reference, prediction)
            errors = result.substitutions + result.deletions + result.insertions
            count = result.hits + result.substitutions + result.deletions
            rows.append({"variant": variant, "region": region["id"], "reference": reference,
                         "prediction": prediction, "edits": errors, "characters": count,
                         "cer": errors/count})
            edits += errors
            characters += count
        summary.append({"variant": variant, "regions": len(annotations["regions"]),
                        "edits": edits, "characters": characters, "cer": edits/characters})
        print(f"{variant}: {edits}/{characters} edits, CER {edits/characters:.2%}")
    write_csv(args.output, rows)
    write_csv(args.output.with_name(args.output.stem+"_summary.csv"), summary)


if __name__ == "__main__":
    main()
