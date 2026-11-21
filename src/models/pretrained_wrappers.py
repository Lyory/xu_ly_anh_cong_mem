"""Adapters to official repositories; weights are explicitly supplied by the user.

UVDoc is loaded in-process. Legacy DocTr/NAFNet use their own Python environment
and official CLI, avoiding incompatible legacy BasicSR/CUDA dependencies here.
"""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import cv2
import numpy as np
from src.utils import read_rgb, write_rgb


class UVDocWrapper:
    """Nạp UVDoc đã huấn luyện và nắn trang ở độ phân giải của ảnh đầu vào."""

    def __init__(self, repo, checkpoint, device="cpu"):
        """Khởi tạo kiến trúc gốc và nạp trọng số trên thiết bị được chọn."""
        import torch
        self.torch, self.device = torch, torch.device(device)
        source = Path(repo).resolve() / "model.py"
        spec = importlib.util.spec_from_file_location("_doc_restore_uvdoc", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.UVDocnet(num_filter=32, kernel_size=5).to(self.device)
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model_state"], strict=True)
        self.model.eval()

    def __call__(self, img):
        """Trả ảnh RGB đã nắn và lưu lưới tọa độ nguồn trong last_bmap."""
        torch = self.torch
        from torch.nn import functional as F
        source = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()[None].to(self.device)/255
        small = cv2.resize(img.astype(np.float32)/255, (488, 712))
        tensor = torch.from_numpy(small.transpose(2, 0, 1).copy())[None].to(self.device)
        with torch.inference_mode():
            points, _ = self.model(tensor)
            grid = F.interpolate(points, img.shape[:2], mode="bilinear", align_corners=True).permute(0, 2, 3, 1)
            # Match upstream UVDoc's zero padding exactly.
            output = F.grid_sample(source, grid, align_corners=True)
            self.last_bmap = grid[0].cpu().numpy()
        return (output[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)*255).round().astype(np.uint8)


class NAFNetWrapper:
    tasks = ("deblur",)

    def __init__(self, repo, config, python="python", timeout=300):
        self.repo, self.config = Path(repo).resolve(), str(Path(config).resolve())
        self.python, self.timeout = python, timeout

    def __call__(self, img):
        with tempfile.TemporaryDirectory(prefix="docrestore_naf_") as directory:
            source, target = Path(directory)/"input.png", Path(directory)/"output.png"
            write_rgb(source, img)
            subprocess.run([self.python, "basicsr/demo.py", "-opt", self.config,
                            "--input_path", str(source), "--output_path", str(target)],
                           cwd=self.repo, check=True, timeout=self.timeout, capture_output=True)
            return read_rgb(target)


class DocTrWrapper:
    def __init__(self, repo, segmentation_checkpoint, geometry_checkpoint, python="python", timeout=300):
        self.repo, self.python, self.timeout = Path(repo).resolve(), python, timeout
        self.seg = str(Path(segmentation_checkpoint).resolve())
        self.geo = str(Path(geometry_checkpoint).resolve())

    def __call__(self, img):
        # Upstream inference initializes IllTr even without --ill_rec. Its default
        # IllTr checkpoint must exist in the repository. Requires upstream CUDA.
        with tempfile.TemporaryDirectory(prefix="docrestore_doctr_") as directory:
            root = Path(directory)
            source, target = root/"input", root/"geo"
            source.mkdir()
            write_rgb(source/"page.png", img)
            subprocess.run([self.python, "inference.py", "--distorrted_path", str(source)+"/",
                            "--gsave_path", str(target)+"/", "--isave_path", str(root/"ill")+"/",
                            "--Seg_path", self.seg, "--GeoTr_path", self.geo],
                           cwd=self.repo, check=True, timeout=self.timeout, capture_output=True)
            return read_rgb(target/"page_geo.png")


def load_backends(config_file):
    import yaml
    config = yaml.safe_load(Path(config_file).read_text()) or {}
    constructors = {"uvdoc": UVDocWrapper, "doctr": DocTrWrapper, "nafnet": NAFNetWrapper}
    result = {}
    for stage, settings in config.items():
        if stage not in {"dewarp_backend", "restore_backend"}:
            raise ValueError(f"Unknown backend stage: {stage}")
        options = dict(settings)
        name = options.pop("name")
        result[stage] = constructors[name](**options)
    return result
