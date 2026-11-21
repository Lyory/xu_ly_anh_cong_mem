# Kết quả kiểm chứng chương trình — 21/11/2026

Đây là smoke test bằng tài liệu tự sinh tiếng Anh. Chưa phải benchmark độc lập và không chứng minh mô hình DL đã học khôi phục tài liệu.

## Các kiểm tra đã chạy

- **34 kiểm thử tự động đạt**: map/đơn vị tọa độ, góc deskew, contour, bộ lọc, metrics, gradient hai mạng, checkpoint, chia tài liệu và OCR/PDF.
- Bốn notebook được thực thi bằng nbclient với kernel `.venv`; bản có output nằm ở `experiments/smoke/executed/`.
- 12 ảnh nguồn tự sinh; mỗi task gồm train=10, val=1, test=1; mỗi ảnh có một biến thể 128×128.
- U-Net base=4 và DewarpNet ResNet18: một batch train, một batch validation, input 64×64 trên CPU. Checkpoint gắn `smoke_only=true`.
- Render PDF hai trang; Gradio HTTP 200, callback phục hồi/ảnh PNG và callback OCR/PDF đã chạy thành công. Server kiểm thử đã được đóng.
- Tesseract 5.3.4 cục bộ; có `eng`, `vie`, `osd`. OCR với `vie+eng` trên nội dung tiếng Anh và PDF có text layer (801 ký tự trích xuất trong mẫu demo). Chưa kiểm chứng độ chính xác trên văn bản tiếng Việt thực.
- `pip check`: không có dependency bị thiếu/xung đột trong môi trường kiểm chứng.

## Ví dụ pipeline bóng/mờ trên một mẫu test

| Cấu hình | PSNR (dB) | SSIM | Ghi chú |
|---|---:|---:|---|
| input | 5.798 | 0.4922 | Baseline trên ảnh demo |
| classical | 22.693 | 0.6606 | Baseline trên ảnh demo |
| dl | 5.799 | 0.4927 | Checkpoint chỉ train một batch |
| hybrid | 5.799 | 0.4927 | Checkpoint chỉ train một batch |

Không suy ra xếp hạng mô hình từ bảng này. DL gần ảnh input vì mới cập nhật một batch; cần train đầy đủ và đánh giá test đa dạng.

## Bộ lọc và OCR

- `denoise_comparison.csv`: 105 hàng = 5 ảnh × 3 kiểu nhiễu × (6 bộ lọc + input).
- `edge_comparison.csv`: 60 hàng = 5 ảnh × 2 điều kiện bóng × 6 bộ lọc biên.
- `edge_summary.csv`: tỉ lệ tìm được contour và sai số góc trên những ảnh tìm được.
- `smoke_ablation.csv`: bảy cấu hình cổ điển; `smoke_hybrid_ablation.csv`: năm cấu hình với khối U-Net hợp nhất.
- `smoke_warp.csv`: PSNR/SSIM/MS-SSIM và map endpoint error cho bốn cấu hình.

| Cấu hình OCR | CER | WER |
|---|---:|---:|
| input | 0.00255 | 0.02083 |
| classical | 0.00255 | 0.02083 |

Ảnh OCR là trang sạch tự sinh, không phải ảnh chụp khó. Bảng OCR không có GT ảnh sạch nên các cột PSNR/SSIM để trống đúng theo thiết kế.

## Artifact và tái lập

- Config/manifest/checkpoint thử: `experiments/smoke/` (không đưa vào Git).
- Log epoch: `experiments/log.csv`; phiên bản thư viện: `experiments/environment.txt`.
- Hình minh họa: `../figures/fft_comparison.png`, `denoise_comparison.png`, `edge_comparison.png`, `pipeline_metrics.png` và thư mục `filters/`.
- CSV cùng JSON cấu hình nằm ngay tại thư mục bảng này.
- Chạy kiểm thử: `source .venv/bin/activate` rồi `OMP_NUM_THREADS=2 python -m pytest -q`.

## Phần cần dữ liệu và tài nguyên thực nghiệm

Thu thập tài liệu sạch đa dạng và 100–150 ảnh chụp thật, train đầy đủ trên GPU, tải pretrained weights đúng giấy phép, rồi chạy lại đánh giá và ablation với text chuẩn tiếng Việt. Adapter UVDoc/NAFNet/DocTr chưa được thử bằng pretrained weights thật. Không thay kết quả kiểm chứng chương trình thành tuyên bố chất lượng phục hồi.
