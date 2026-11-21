import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import random
import fcntl
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from src.data.datasets import PairedDataset
from src.models.unet_attention import UNetAttention
from src.models.dewarp_net import DewarpNet
from src.losses import CombinedLoss, DewarpLoss


def train(task):
    parser = argparse.ArgumentParser(description=f"Train {task}")
    parser.add_argument("--config", type=Path, default=Path("configs/dewarp.yaml" if task == "warp" else "configs/shadow_deblur.yaml"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-batches", type=int, help="Smoke checks only; limits both train and validation")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if config["task"] != task:
        parser.error("Config task mismatch")
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["image_size"] < 64:
        parser.error("epochs/batch_size >=1 and image_size >=64 required")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("max-batches must be >=1")
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device_name = config.get("device", "auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    output = Path(config["output_dir"])
    if (output / "last.pt").exists() and not args.resume:
        parser.error("Output already contains a run; use --resume or another output_dir")
    output.mkdir(parents=True, exist_ok=True)
    loaders = {}
    generator = torch.Generator().manual_seed(config["seed"])
    for split in ["train", "val"]:
        dataset = PairedDataset(config["data_root"], split, task, config["image_size"], config["model"].get("in_ch", 3))
        loaders[split] = DataLoader(dataset, batch_size=config["batch_size"], shuffle=split == "train",
                                    num_workers=config["num_workers"], generator=generator, pin_memory=device.type == "cuda")
    model = (DewarpNet(**config["model"]) if task == "warp" else UNetAttention(**config["model"])).to(device)
    criterion = (DewarpLoss(**config["loss"]) if task == "warp" else CombinedLoss(**config["loss"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    amp = bool(config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best = 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint["task"] != task or checkpoint["config"]["model"] != config["model"]:
            parser.error("Checkpoint architecture/task mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, best = checkpoint["epoch"]+1, checkpoint["best_val"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        generator.set_state(checkpoint["loader_rng"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng"):
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
    (output / "config.yaml").write_text(yaml.safe_dump(config))
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(output / "tensorboard")) as writer:
        for epoch in range(start, config["epochs"]):
            values = {}
            for split, loader in loaders.items():
                model.train(split == "train")
                total, count = 0., 0
                for i, batch in enumerate(loader):
                    if args.max_batches and i >= args.max_batches:
                        break
                    source, target = batch["input"].to(device), batch["target"].to(device)
                    with torch.set_grad_enabled(split == "train"), torch.autocast(device.type, enabled=amp):
                        prediction = model(source)
                        loss = criterion(prediction, source, target, batch["bmap"].to(device)) if task == "warp" else criterion(prediction, target)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss in {split}, epoch {epoch}")
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        scaler.step(optimizer)
                        scaler.update()
                    total += loss.item()*len(source)
                    count += len(source)
                values[split] = total/count
                writer.add_scalar(f"loss/{split}", values[split], epoch)
            scheduler.step()
            improved = values["val"] < best
            best = min(best, values["val"])
            checkpoint = {"task": task, "model": model.state_dict(), "config": config,
                          "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                          "scaler": scaler.state_dict(), "epoch": epoch, "best_val": best,
                          "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
                          "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                          "smoke_only": args.max_batches is not None}
            torch.save(checkpoint, output / "last.pt")
            if improved:
                torch.save(checkpoint, output / "best.pt")
            log = Path("experiments/log.csv")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", newline="") as stream:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                write_header = log.stat().st_size == 0
                row = {"timestamp": datetime.now(timezone.utc).isoformat(), "task": task,
                       "run": str(output), "epoch": epoch+1, "train_loss": values["train"],
                       "val_loss": values["val"], "smoke_only": args.max_batches is not None}
                csv_writer = csv.DictWriter(stream, fieldnames=list(row))
                if write_header:
                    csv_writer.writeheader()
                csv_writer.writerow(row)
                stream.flush()
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            print(f"Epoch {epoch+1}: train={values['train']:.5f}, val={values['val']:.5f}", flush=True)
