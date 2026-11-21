from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from src.utils import image_files, read_rgb
from src.classical.shadow_removal import estimate_background


def to_tensor(image):
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255


def model_input(image, in_ch=3):
    tensor = to_tensor(image)
    if in_ch == 4:
        background = estimate_background(image)
        gray = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
        tensor = torch.cat([tensor, torch.from_numpy(gray.copy()).float().unsqueeze(0)/255])
    elif in_ch != 3:
        raise ValueError("in_ch must be 3 or 4")
    return tensor


class PairedDataset(Dataset):
    def __init__(self, root, split="train", task="shadow_deblur", size=None, in_ch=3):
        self.root = Path(root) / split
        self.files = image_files(self.root / "input")
        self.task, self.size, self.in_ch = task, size, in_ch
        if task not in {"shadow_deblur", "warp"}:
            raise ValueError(f"Unknown task: {task}")
        if not self.files:
            raise ValueError(f"No input images in {self.root}")
        for path in self.files:
            relative = path.relative_to(self.root / "input")
            if not (self.root / "target" / relative).is_file():
                raise FileNotFoundError(f"Missing target for {path}")
            if task == "warp" and not (self.root / "bmap" / relative.with_suffix(".npy")).is_file():
                raise FileNotFoundError(f"Missing bmap for {path}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        relative = path.relative_to(self.root / "input")
        source, target = read_rgb(path), read_rgb(self.root / "target" / relative)
        original_shape = source.shape[:2]
        if source.shape != target.shape:
            raise ValueError(f"Unaligned pair: {path}")
        if self.size:
            source = cv2.resize(source, (self.size, self.size), interpolation=cv2.INTER_AREA)
            target = cv2.resize(target, (self.size, self.size), interpolation=cv2.INTER_AREA)
        result = {"input": model_input(source, self.in_ch), "target": to_tensor(target), "id": path.stem}
        if self.task == "warp":
            grid = np.load(self.root / "bmap" / relative.with_suffix(".npy"), allow_pickle=False)
            if grid.shape != (*original_shape, 2) or not np.isfinite(grid).all():
                raise ValueError(f"Invalid bmap for {path}")
            if self.size:
                grid = cv2.resize(grid, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            result["bmap"] = torch.from_numpy(grid.copy()).float()  # BH W 2 after batching
        return result
