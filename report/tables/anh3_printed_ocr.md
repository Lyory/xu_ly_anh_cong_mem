# CER trên phần chữ in của anh3.jpg

Phép đo dùng 16 vùng chữ in tự phiên âm trên cùng một tờ biểu mẫu, tổng cộng
284 ký tự chuẩn sau chuẩn hóa. Có vùng ở tiêu đề, thân biểu mẫu và nếp gấp vùng
“Outside”. Các vùng chữ viết tay được loại khỏi phép đo. Ảnh gốc và ảnh quét
dùng cùng nội dung và tọa độ cắt riêng. Toàn bộ ba kiểu xuất ảnh chia sẻ cùng
bước chỉnh hình học.

| Đầu vào Tesseract | Số lỗi / ký tự chuẩn | CER |
|---|---:|---:|
| Ảnh chụp gốc | 26 / 284 | 9,15% |
| Magic Color | 8 / 284 | 2,82% |
| Đen trắng mềm | 7 / 284 | 2,46% |
| Nhị phân Sauvola | 9 / 284 | 3,17% |

Tesseract 5.3.4 dùng mô hình `eng` và `--psm 7` cho từng dòng. Chuẩn hóa
NFKC, chữ thường, các dấu gạch ngang và khoảng trắng trước khi tính khoảng cách
chỉnh sửa. CER gộp bằng tổng số thay, xóa và chèn ký tự chia tổng độ dài chuẩn.
Các vùng cắt tay giúp cô lập chất lượng đọc chữ in. Mức CER chỉ mô tả các dòng
đã chọn của một ảnh và còn lỗi ở hai vùng qua nếp gấp. Độ chính xác OCR toàn
trang và chữ viết tay cần phép đo riêng.

Xem [tọa độ và văn bản chuẩn](../annotations/anh3_printed.json),
[kết quả từng vùng](anh3_printed_ocr.csv) và [CSV tổng hợp](anh3_printed_ocr_summary.csv).

Tạo ba ảnh quét bằng lệnh trong [README](../../README.md), rồi chạy từ thư mục gốc.

```bash
source .venv/bin/activate
python -m scripts.evaluate_printed_ocr \
  --regions report/annotations/anh3_printed.json \
  --restored outputs/scan/anh3_magic-color.png outputs/scan/anh3_soft-bw.png outputs/scan/anh3_sauvola.png \
  --output report/tables/anh3_printed_ocr.csv
```
