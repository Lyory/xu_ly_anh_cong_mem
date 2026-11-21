from pathlib import Path
import shutil


def _engine(lang):
    """Kiểm tra Tesseract và các gói ngôn ngữ cần cho tài liệu."""
    import pytesseract
    if not shutil.which("tesseract"):
        raise RuntimeError("Install Tesseract system executable and language packs (vie, eng)")
    missing = set(lang.split("+")) - set(pytesseract.get_languages(config=""))
    if missing:
        raise RuntimeError(f"Missing Tesseract language packs: {sorted(missing)}")
    return pytesseract


def recognize(img, lang="vie+eng", psm=6):
    """Đọc văn bản bằng Tesseract với kiểu phân vùng trang đã chọn."""
    return _engine(lang).image_to_string(img, lang=lang, config=f"--psm {int(psm)}")


def searchable_pdf(img, output, lang="vie+eng", psm=6):
    """Ghi ảnh trang cùng lớp chữ OCR vào PDF có thể tìm kiếm."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _engine(lang).image_to_pdf_or_hocr(img, extension="pdf", lang=lang,
                                               config=f"--psm {int(psm)}")
    path.write_bytes(content)
    return path
