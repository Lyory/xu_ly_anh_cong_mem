import argparse
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Render PDF pages into grouped clean RGB images")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/raw_clean"))
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    import pymupdf
    if args.dpi <= 0:
        parser.error("dpi must be positive")
    pdfs = sorted(args.input.rglob("*.pdf")) if args.input.is_dir() else [args.input]
    if not pdfs:
        parser.error("No PDF files found")
    args.output.mkdir(parents=True, exist_ok=True)
    for pdf in pdfs:
        digest = hashlib.sha256(pdf.read_bytes()).hexdigest()[:10]
        with pymupdf.open(pdf) as document:
            for index, page in enumerate(document):
                pixmap = page.get_pixmap(dpi=args.dpi, colorspace=pymupdf.csRGB, alpha=False)
                pixmap.save(args.output / f"{pdf.stem}_{digest}__p{index+1:04d}.png")
    print(f"Rendered {len(pdfs)} PDFs to {args.output}")


if __name__ == "__main__":
    main()
