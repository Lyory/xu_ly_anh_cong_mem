import argparse
from pathlib import Path
import tempfile
import gradio as gr
from src.pipeline import DocRestorer
from src.classical.preprocess import load_image
from src.utils import write_rgb


def build_demo(shadow_checkpoint=None, dewarp_checkpoint=None, device="cpu", backend_config=None):
    def restore(path, mode, adaptive, style, denoiser, deskew_method, do_ocr, lang):
        if path is None:
            raise gr.Error("Hãy tải ảnh tài liệu lên.")
        try:
            # Giao diện cũ gửi trạng thái ô “Nhị phân hóa” tại vị trí của kiểu ảnh.
            if isinstance(style, bool):
                style = "sauvola" if style else "magic-color"
            backends = {}
            neural = mode in {"dl", "hybrid"}
            if backend_config and neural:
                from src.models.pretrained_wrappers import load_backends
                backends = load_backends(backend_config)
            restorer = DocRestorer(mode, shadow_checkpoint if neural else None,
                                  dewarp_checkpoint if neural else None, device,
                                  adaptive=adaptive, scan_style=style,
                                  max_size=2600 if mode == "scan" else 1800,
                                  denoise_method=denoiser, deskew_method=deskew_method, **backends)
            result = restorer.restore(load_image(path))
            text, pdf = "", None
            run_dir = Path(tempfile.mkdtemp(prefix="docrestore_demo_"))
            png = run_dir / "restored.png"
            write_rgb(png, result["image"])
            if do_ocr:
                from src.ocr import recognize, searchable_pdf
                text = recognize(result["image"], lang)
                pdf = str(searchable_pdf(result["image"], run_dir/"searchable.pdf", lang))
            gallery = [(image, name) for name, image in result["intermediates"].items()]
            return result["image"], gallery, result["metadata"], text, str(png), pdf
        except (ValueError, RuntimeError, OSError, ImportError) as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="Khôi phục ảnh tài liệu") as demo:
        gr.Markdown("# Khôi phục ảnh tài liệu\nNắn phối cảnh, chỉnh nghiêng, giảm cong, khử bóng, nhiễu và mờ.")
        with gr.Row():
            image = gr.Image(type="filepath", label="Ảnh chụp tài liệu")
            output = gr.Image(type="numpy", label="Ảnh sau xử lý")
        with gr.Row():
            mode = gr.Dropdown([("Quét tài liệu màu", "scan"), ("Cổ điển", "classical"),
                                ("Học sâu", "dl"), ("Kết hợp", "hybrid")],
                               value="scan", label="Phương pháp")
            denoiser = gr.Dropdown(["mean", "gaussian", "median", "bilateral", "nlm", "wiener"], value="bilateral", label="Khử nhiễu")
            skew = gr.Dropdown(["hough", "projection", "min_area_rect"], value="hough", label="Chỉnh nghiêng")
        with gr.Row():
            adaptive = gr.Checkbox(value=False, label="Tự chọn bước theo chẩn đoán")
            style = gr.Dropdown([("Giữ màu (Magic Color)", "magic-color"), ("Đen trắng mềm", "soft-bw"),
                                 ("Nhị phân Sauvola", "sauvola")], value="magic-color", label="Kiểu ảnh xuất",
                                allow_custom_value=True)
            do_ocr = gr.Checkbox(value=False, label="Nhận dạng chữ và tạo PDF")
            lang = gr.Dropdown(["vie+eng", "eng", "vie"], value="vie+eng", label="Ngôn ngữ")
        button = gr.Button("Khôi phục", variant="primary")
        gallery = gr.Gallery(label="Các bước xử lý", columns=3)
        metadata = gr.JSON(label="Chẩn đoán và thời gian xử lý")
        text = gr.Textbox(label="Nội dung OCR", lines=8)
        with gr.Row():
            download = gr.File(label="Tải ảnh PNG")
            pdf = gr.File(label="Tải PDF có thể tìm kiếm")
        button.click(restore, [image, mode, adaptive, style, denoiser, skew, do_ocr, lang],
                     [output, gallery, metadata, text, download, pdf])
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow-checkpoint")
    parser.add_argument("--dewarp-checkpoint")
    parser.add_argument("--backend-config")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_demo(args.shadow_checkpoint, args.dewarp_checkpoint, args.device, args.backend_config).queue().launch(
        server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
