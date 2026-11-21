from pathlib import Path
import time
import cv2
import numpy as np
from src.classical.preprocess import preprocess, load_image
from src.classical.detect_document import detect_document, rectify_document
from src.classical.dewarp_classic import dewarp_classic
from src.classical.deskew import deskew
from src.classical.shadow_removal import remove_shadow
from src.classical.scan import rectify_paper, whiten_paper
from src.classical.denoise import denoise
from src.classical.deblur import deblur, unsharp_mask
from src.classical.binarize import SCAN_STYLES, binarize, clahe, render_scan_style
from src.diagnose import diagnose, select_modules
from src.utils import validate_rgb

DEFAULT_ORDER = ("perspective", "deskew", "dewarp", "shadow", "denoise", "deblur", "restore_dl", "contrast", "binarize")
LOCAL_UVDOC = Path(__file__).resolve().parents[1] / "external" / "UVDoc"


def _scan_dewarper(device):
    """Tạo hàm nắn UVDoc khi tìm thấy mã mô hình và trọng số cục bộ."""
    checkpoint = LOCAL_UVDOC / "model" / "best_model.pkl"
    if not ((LOCAL_UVDOC / "model.py").is_file() and checkpoint.is_file()):
        return None

    wrapper = None

    def dewarp(image):
        """Nạp UVDoc ở lần gọi đầu và trả ảnh đã nắn cùng lưới lấy mẫu ngược."""
        nonlocal wrapper
        if wrapper is None:
            from src.models.pretrained_wrappers import UVDocWrapper
            wrapper = UVDocWrapper(LOCAL_UVDOC, checkpoint, device)
        return wrapper(image), wrapper.last_bmap

    return dewarp


class DocRestorer:
    """Điều phối các bước phục hồi theo chế độ đã chọn."""
    def __init__(self, mode="classical", shadow_checkpoint=None, dewarp_checkpoint=None,
                 device="cpu", modules=None, order=None, adaptive=False, max_size=1800,
                 white_balance=False, edge_method="canny", deskew_method="hough",
                 shadow_method="divide", denoise_method="bilateral", deblur_method="unsharp",
                 estimate_psf=False, dewarp_backend=None, restore_backend=None, neural_size=512,
                 scan_style=None):
        """Chọn mô hình, thứ tự xử lý và các bước bật cho chế độ phục hồi."""
        if mode not in {"classical", "scan", "dl", "hybrid"}:
            raise ValueError(f"Unknown mode: {mode}")
        if neural_size < 64:
            raise ValueError("neural_size must be >=64")
        if scan_style is not None and scan_style not in SCAN_STYLES:
            raise ValueError(f"Unknown scan style: {scan_style}")
        self.scan_style = scan_style
        self.mode, self.device = mode, device
        self.adaptive, self.max_size, self.white_balance = adaptive, max_size, white_balance
        self.edge_method, self.deskew_method = edge_method, deskew_method
        self.shadow_method, self.denoise_method, self.deblur_method = shadow_method, denoise_method, deblur_method
        self.estimate_psf, self.neural_size = estimate_psf, neural_size
        self.dewarp_backend, self.restore_backend = dewarp_backend, restore_backend
        self.shadow_model = self.dewarp_model = None
        self.in_ch = 3
        self.model_metadata = {}
        if shadow_checkpoint and restore_backend:
            raise ValueError("Choose one restoration checkpoint or backend")
        if dewarp_checkpoint and dewarp_backend:
            raise ValueError("Choose one dewarp checkpoint or backend")
        if mode in {"classical", "scan"} and any([shadow_checkpoint, dewarp_checkpoint, dewarp_backend, restore_backend]):
            raise ValueError("Neural checkpoints/backends require dl or hybrid mode")
        if shadow_checkpoint:
            self.shadow_model, config = self._load(shadow_checkpoint, "shadow_deblur")
            self.in_ch = config["model"].get("in_ch", 3)
        if dewarp_checkpoint:
            self.dewarp_model, _ = self._load(dewarp_checkpoint, "warp")
        neural_restore = self.shadow_model is not None or restore_backend is not None
        neural_dewarp = self.dewarp_model is not None or dewarp_backend is not None
        restore_tasks = {"shadow", "deblur"} if self.shadow_model is not None else set(getattr(restore_backend, "tasks", ("shadow", "deblur")))
        if mode == "dl" and not (neural_restore or neural_dewarp):
            raise ValueError("DL mode requires a trained checkpoint or configured pretrained backend")
        self.modules = dict.fromkeys(DEFAULT_ORDER, True)
        self.modules.update({"binarize": False, "contrast": False, "restore_dl": neural_restore,
                             "shadow": not (neural_restore and "shadow" in restore_tasks) and mode != "dl",
                             "deblur": not (neural_restore and "deblur" in restore_tasks) and mode != "dl",
                             "dewarp": neural_dewarp or mode != "dl"})
        if mode == "scan":
            self.modules.update({"deskew": False, "dewarp": False, "denoise": False})
        if modules:
            if set(modules)-set(DEFAULT_ORDER):
                raise ValueError(f"Unknown modules: {set(modules)-set(DEFAULT_ORDER)}")
            self.modules.update(modules)
        if neural_restore and any(self.modules[stage] for stage in restore_tasks):
            raise ValueError("Joint U-Net/pretrained restoration uses restore_dl; disable classical shadow/deblur")
        if mode == "dl" and self.modules["dewarp"] and not neural_dewarp:
            raise ValueError("DL dewarp requires dewarp checkpoint/backend")
        self.order = tuple(order or DEFAULT_ORDER)
        if len(set(self.order)) != len(self.order) or set(self.order) != set(DEFAULT_ORDER):
            raise ValueError("order must be a permutation of all pipeline stages")

    def _load(self, path, task):
        """Nạp trọng số đúng với tác vụ và chuyển mô hình sang chế độ suy luận."""
        import torch
        from src.models.unet_attention import UNetAttention
        from src.models.dewarp_net import DewarpNet
        if not Path(path).is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if checkpoint["task"] != task:
            raise ValueError(f"Expected {task} checkpoint")
        config = checkpoint["config"]
        model_options = dict(config["model"])
        if task == "warp":
            model_options["pretrained_encoder"] = False  # Saved weights already include the encoder.
        model = (DewarpNet(**model_options) if task == "warp" else UNetAttention(**model_options)).to(self.device)
        model.load_state_dict(checkpoint["model"], strict=True)
        self.model_metadata[task] = {"checkpoint": str(path), "epoch": checkpoint["epoch"],
                                     "smoke_only": checkpoint.get("smoke_only", False)}
        return model.eval(), config

    def _neural(self, img, task):
        """Dự đoán lưới nắn hoặc phần hiệu chỉnh màu rồi áp dụng ở độ phân giải ảnh."""
        import torch
        from torch.nn import functional as F
        from src.data.datasets import model_input, to_tensor
        from src.models.dewarp_net import unwarp, flow_to_grid
        h, w = img.shape[:2]
        scale = min(1., self.neural_size/max(h, w))
        size = (max(64, round(w*scale)), max(64, round(h*scale)))
        small = cv2.resize(img, size)
        with torch.inference_mode():
            if task == "warp":
                flow = self.dewarp_model(to_tensor(small)[None].to(self.device))
                flow = F.interpolate(flow, (h, w), mode="bilinear", align_corners=True)
                output = unwarp(to_tensor(img)[None].to(self.device), flow)
                grid = flow_to_grid(flow)[0].cpu().numpy()
            else:
                tensor = model_input(small, self.in_ch)[None].to(self.device)
                prediction = self.shadow_model(tensor)
                # Upsample the residual to retain original-resolution text detail.
                residual = F.interpolate(prediction-tensor[:, :3], (h, w), mode="bilinear", align_corners=False)
                output = to_tensor(img)[None].to(self.device)+residual
                grid = None
        image = (output[0].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)*255).round().astype(np.uint8)
        return image, grid

    def restore(self, img):
        """Nhận đường dẫn hoặc ảnh RGB, trả ảnh cuối, ảnh trung gian và thông số xử lý."""
        started = time.perf_counter()
        if isinstance(img, (str, Path)):
            img = load_image(img)
        validate_rgb(img)
        current = preprocess(img, self.max_size, self.white_balance)
        scores = diagnose(current)
        enabled = self.modules.copy()
        if self.adaptive:
            selected = select_modules(scores)
            for name in ("shadow", "deblur", "dewarp"):
                enabled[name] = enabled[name] and selected[name]
            enabled["restore_dl"] = enabled["restore_dl"] and (selected["shadow"] or selected["deblur"])
        intermediates = {"input": img.copy(), "preprocess": current.copy()}
        metadata = {"mode": self.mode, "scores": scores, "enabled": enabled, "models": self.model_metadata,
                    "order": list(self.order), "warnings": [], "timings": {}}
        style = self.scan_style
        if self.mode == "scan" and style is None:
            style = "sauvola" if enabled["binarize"] else "magic-color"
        bmap = None
        for stage in self.order:
            if not enabled[stage]:
                continue
            tick = time.perf_counter()
            if stage == "perspective":
                if self.mode == "scan":
                    page, geometry = rectify_paper(current, _scan_dewarper(self.device))
                    if page is not None:
                        current = page
                        metadata["paper_geometry"] = geometry
                    else:
                        metadata["warnings"].append("No confident paper outline; crop skipped")
                else:
                    corners = detect_document(current, edge_method=self.edge_method)
                    if corners is not None:
                        current, matrix = rectify_document(current, corners)
                        metadata["perspective_matrix"] = matrix.tolist()
                    else:
                        metadata["warnings"].append("No confident document quadrilateral; perspective skipped")
            elif stage == "deskew":
                current, metadata["deskew_angle"] = deskew(current, self.deskew_method)
            elif stage == "dewarp":
                if self.dewarp_backend:
                    current = self.dewarp_backend(current)
                    bmap = getattr(self.dewarp_backend, "last_bmap", None)
                elif self.dewarp_model is not None:
                    current, bmap = self._neural(current, "warp")
                else:
                    current, bmap = dewarp_classic(current, return_map=True)
            elif stage == "shadow":
                if self.mode == "scan":
                    table = metadata.get("paper_geometry", {}).get("table_rectification", {})
                    current = whiten_paper(current, table.get("table_lower_edge"))
                else:
                    current = remove_shadow(current, self.shadow_method)
            elif stage == "denoise":
                current = denoise(current, self.denoise_method)
            elif stage == "deblur":
                psf = None
                if self.estimate_psf and self.deblur_method != "unsharp":
                    from src.classical.frequency import estimate_motion_psf
                    psf, metadata["psf"] = estimate_motion_psf(current)
                if self.mode == "scan" and self.deblur_method == "unsharp":
                    current = unsharp_mask(current, amount=.28)
                else:
                    current = deblur(current, self.deblur_method, psf=psf)
            elif stage == "restore_dl":
                if self.restore_backend:
                    current = self.restore_backend(current)
                elif self.shadow_model is not None:
                    current, _ = self._neural(current, "shadow_deblur")
                else:
                    raise ValueError("restore_dl requires a checkpoint/backend")
            elif stage == "contrast":
                current = clahe(current)
            elif stage == "binarize":
                if style is not None:
                    continue
                # Light median filtering suppresses isolated threshold artifacts.
                current = binarize(denoise(current, "median"))
            validate_rgb(current)
            intermediates[stage] = current.copy()
            metadata["timings"][stage] = time.perf_counter()-tick
        if style is not None:
            tick = time.perf_counter()
            current = render_scan_style(current, style, clean_edges=bool(metadata.get("paper_geometry")))
            intermediates["scan_style"] = current.copy()
            metadata["scan_style"] = style
            metadata["timings"]["scan_style"] = time.perf_counter()-tick
        metadata["elapsed_seconds"] = time.perf_counter()-started
        return {"image": current, "intermediates": intermediates, "metadata": metadata, "bmap": bmap}
