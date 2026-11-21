import shutil
import cv2
import numpy as np
import pytest
from src.metrics import text_metrics


def test_text_error_rates():
    pytest.importorskip("jiwer")
    assert text_metrics("hello world", "hello world") == {"cer": 0., "wer": 0.}
    assert text_metrics("hello word", "hello world")["wer"] == pytest.approx(.5)
    assert text_metrics("abc", "") == {"cer": 3., "wer": 1.}


@pytest.mark.skipif(not shutil.which("tesseract"), reason="Tesseract executable not installed")
def test_ocr_searchable_pdf_roundtrip(tmp_path):
    pytest.importorskip("pytesseract")
    pymupdf = pytest.importorskip("pymupdf")
    from src.ocr import recognize, searchable_pdf
    image = np.full((220, 800, 3), 255, np.uint8)
    cv2.putText(image, "DOCUMENT RESTORATION", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 2, cv2.LINE_AA)
    assert "DOCUMENT" in recognize(image, "eng").upper()
    pdf = searchable_pdf(image, tmp_path/"searchable.pdf", "eng")
    with pymupdf.open(pdf) as document:
        assert len(document) == 1
        assert "DOCUMENT" in document[0].get_text().upper()
