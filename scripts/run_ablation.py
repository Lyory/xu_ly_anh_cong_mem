import argparse
from pathlib import Path
from scripts.evaluate import evaluate
from src.pipeline import DEFAULT_ORDER, DocRestorer


def main():
    parser = argparse.ArgumentParser(description="Module removal and ordering ablation, same input/GT")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--bmap", type=Path)
    parser.add_argument("--text", type=Path)
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--lang", default="vie+eng")
    parser.add_argument("--mode", choices=["classical", "dl", "hybrid"], default="classical")
    parser.add_argument("--shadow-checkpoint")
    parser.add_argument("--dewarp-checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("report/tables/ablation.csv"))
    args = parser.parse_args()
    if args.text and not args.ocr:
        parser.error("--text requires --ocr")
    base = {"mode": args.mode, "shadow_checkpoint": args.shadow_checkpoint,
            "dewarp_checkpoint": args.dewarp_checkpoint, "device": args.device}
    restorer = DocRestorer(**base)
    configs = [{"name": "full", "options": base}]
    for stage in ["dewarp", "shadow", "denoise", "deblur", "restore_dl"]:
        if restorer.modules[stage]:
            configs.append({"name": f"without_{stage}", "options": {**base, "modules": {stage: False}}})
    for name, a, b in [("deblur_before_denoise", "deblur", "denoise"),
                       ("shadow_before_dewarp", "shadow", "dewarp")]:
        if restorer.modules[a] and restorer.modules[b]:
            order = list(DEFAULT_ORDER)
            ia, ib = order.index(a), order.index(b)
            order[ia], order[ib] = order[ib], order[ia]
            configs.append({"name": name, "options": {**base, "order": order}})
    if restorer.modules["restore_dl"]:
        order = list(DEFAULT_ORDER)
        i, j = order.index("restore_dl"), order.index("denoise")
        order[i], order[j] = order[j], order[i]
        configs.append({"name": "restore_dl_before_denoise", "options": {**base, "order": order}})
    rows = evaluate(args.input, args.output, configs, args.target, args.bmap, args.text, args.ocr, args.lang)
    print(f"Saved {len(rows)} ablation rows to {args.output}")


if __name__ == "__main__":
    main()
