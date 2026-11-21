import argparse
import json
from pathlib import Path
from src.pipeline import DocRestorer
from src.classical.binarize import SCAN_STYLES
from src.utils import write_rgb


def main(default_mode="classical"):
    """Đọc tham số, chạy phục hồi ảnh và ghi các tệp đầu ra được chọn."""
    parser = argparse.ArgumentParser(description="Restore one photographed document")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/restored.png"))
    parser.add_argument("--mode", choices=("classical", "scan", "dl", "hybrid", *SCAN_STYLES), default=default_mode)
    parser.add_argument("--style", choices=SCAN_STYLES, help="Output tone for the scan")
    parser.add_argument("--shadow-checkpoint")
    parser.add_argument("--dewarp-checkpoint")
    parser.add_argument("--backend-config")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--binarize", action="store_true", help="Select Sauvola when using scan mode")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--no-perspective", action="store_true")
    parser.add_argument("--no-deskew", action="store_true")
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--lang", default="vie+eng")
    args = parser.parse_args()
    if args.mode in SCAN_STYLES:
        if args.style is not None:
            parser.error("Select the scan style with either --mode or --style")
        args.style, args.mode = args.mode, "scan"
    if args.style is not None and args.binarize:
        parser.error("Use --style sauvola instead of combining --style with --binarize")
    backends = {}
    if args.backend_config:
        from src.models.pretrained_wrappers import load_backends
        backends = load_backends(args.backend_config)
    modules = {"binarize": args.binarize, "perspective": not args.no_perspective}
    if args.no_deskew:
        modules["deskew"] = False
    restorer = DocRestorer(args.mode, args.shadow_checkpoint, args.dewarp_checkpoint, args.device,
                          modules=modules, adaptive=args.adaptive, scan_style=args.style,
                          max_size=2600 if args.mode == "scan" else 1800, **backends)
    result = restorer.restore(args.input)
    write_rgb(args.output, result["image"])
    for name, image in result["intermediates"].items():
        write_rgb(args.output.parent / f"{args.output.stem}_stages" / f"{name}.png", image)
    args.output.with_suffix(".json").write_text(json.dumps(result["metadata"], indent=2, ensure_ascii=False))
    if args.ocr:
        from src.ocr import recognize
        args.output.with_suffix(".txt").write_text(recognize(result["image"], args.lang), encoding="utf-8")
    if args.pdf:
        from src.ocr import searchable_pdf
        searchable_pdf(result["image"], args.output.with_suffix(".pdf"), args.lang)
    print(args.output)


if __name__ == "__main__":
    main()
