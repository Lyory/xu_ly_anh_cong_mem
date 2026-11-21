# Khôi phục ảnh tài liệu bị cong, bóng đổ và mờ

## 1. Mục tiêu và phạm vi

Xây dựng hệ thống khôi phục khả năng đọc và OCR từ ảnh chụp tài liệu. So sánh
phương pháp xử lý ảnh cổ điển, mạng tự huấn luyện và mô hình pretrained. Đầu ra
gồm ảnh phục hồi, các bước trung gian, văn bản OCR và PDF tìm kiếm được.

Project đã có mã nguồn cho toàn bộ luồng. Chất lượng trên dữ liệu thực chưa được
xác nhận khi chưa thu thập tập ảnh thật, train đầy đủ và tải pretrained weights.
Các số liệu kiểm tra bằng ảnh tự sinh chỉ chứng minh chương trình hoạt động.

## 2. Cơ sở lý thuyết

Mô hình ảnh quan sát xấp xỉ: `I = JPEG(Blur(Warp(R × L)) + n)`, trong đó R là ảnh
phản xạ tài liệu sạch, L là trường chiếu sáng, Warp là biến dạng hình học, n là nhiễu.
Phân rã này chỉ gần đúng: mờ không gian biến thiên và vùng tự che khuất không thể
khôi phục đầy đủ bằng mô hình 2D đơn giản.

### 2.1 Biên và hình học

Sobel/Prewitt xấp xỉ đạo hàm bậc nhất theo hai trục; Scharr giảm bất đẳng hướng.
Magnitude `sqrt(Gx² + Gy²)` biểu thị độ mạnh của chuyển tiếp cường độ. Laplacian
là đạo hàm bậc hai, nhạy với nhiễu; LoG làm trơn Gaussian trước Laplacian. Canny
kết hợp làm trơn, non-maximum suppression và hysteresis để tạo biên mảnh liên kết.
Không mặc định Canny luôn tốt nhất: chọn bằng tỉ lệ tìm được trang và sai số góc
trên cùng tập bóng, sau khi hiệu chỉnh tham số chỉ trên validation.

Contour bốn điểm xác định homography nắn phối cảnh phẳng. HoughLinesP ước lượng
góc nghiêng từ trung vị các đoạn gần ngang; projection profile tối đa hóa phương
sai tổng hàng; minAreaRect dùng hình chữ nhật bao vùng chữ. Deskew là xoay toàn
cục; dewarp xử lý độ cong cục bộ bằng trường tọa độ.

### 2.2 Bóng, nhiễu và mờ

Ước lượng nền bằng dilation và median rồi chia ảnh cho nền giảm biến thiên ánh
sáng chậm. Homomorphic chuyển nhân chiếu sáng thành cộng trong log, loại thành
phần tần số thấp. Hai cách có thể xóa nét nhạt hoặc làm sai màu nếu nền chứa hình.

Mean/Gaussian làm trơn và có thể mất chữ nhỏ. Median bền với xung muối tiêu;
bilateral dùng trọng số không gian và cường độ để giữ biên; NLM tìm patch giống
nhau; Wiener không gian điều chỉnh theo phương sai cục bộ và công suất nhiễu.
Đây là kỳ vọng lý thuyết cần kiểm chứng bằng bảng thực nghiệm, không phải kết
luận rằng NLM/bilateral luôn thắng trên mọi ảnh Gaussian.

FFT biểu diễn năng lượng theo tần số. Gaussian blur làm giảm tần số cao; motion
blur có thể tạo dải triệt tiêu gần vuông góc hướng chuyển động trong phổ. Nội dung
dòng chữ cũng tạo hướng phổ nên ước lượng PSF dễ nhầm. Wiener tần số dùng
`H* / (|H|² + K)`; Richardson–Lucy lặp nhân theo mô hình PSF, nhạy với PSF sai và
nhiễu. Khử nhiễu trước deconvolution giúp hạn chế khuếch đại nhiễu; unsharp là
baseline tăng nét, không phục hồi thông tin đã mất.

### 2.3 Mạng học sâu

U-Net attention dùng bốn mức downsample, skip có attention gate và dự đoán residual
RGB. Loss gồm L1, SSIM, VGG16 đóng băng tùy chọn và sai khác Sobel magnitude.
DewarpNet ResNet18 dự đoán flow chuẩn hóa hai kênh; cộng identity grid và lấy mẫu
bằng `grid_sample(align_corners=True)`. Loss map có supervision tổng hợp, kết hợp
sai số ảnh đã unwarp và độ trơn flow. Không dùng trọng số ngẫu nhiên làm đối chứng.

## 3. Thiết kế thực nghiệm

1. Thu thập tài liệu sạch được phép sử dụng, loại trùng; tách train/val/test theo
   tài liệu nguồn trước khi sinh bóng/mờ/nhiễu/cong. Giữ seed và manifest.
2. Chụp 100–150 ảnh thực: trang phẳng, cuộn sách, bóng mềm/cứng, motion blur,
   nhiều mức nhiễu; chép text chuẩn UTF-8 và ghi thiết bị, ánh sáng, loại suy giảm.
3. Chọn siêu tham số và ngưỡng chẩn đoán trên validation; chỉ đánh giá test cuối.
4. So sánh input, classical, U-Net, DewarpNet, hybrid, UVDoc và NAFNet khi có weight.
5. Ablation bỏ dewarp/shadow/denoise/deblur với classical; bỏ khối restore_dl với
   U-Net hợp nhất; thử đổi thứ tự denoise/deblur, shadow/dewarp.
6. Thí nghiệm bộ lọc dùng cùng ảnh nhiễu cho mọi phương pháp. Báo mean/std và
   thời gian trên cùng máy; contour báo detection rate và corner error, tránh
   chỉ báo lỗi trên những ảnh tìm được biên mà bỏ qua các ca thất bại.

Các lệnh chạy và cấu hình nằm trong README. Hai dataset riêng chưa phải dữ liệu
phối hợp đủ mọi suy giảm; cần bổ sung test tổ hợp trên cùng canvas trước khi
tuyên bố mô hình tổng quát hóa cho ảnh vừa cong vừa bóng vừa mờ.

## 4. Chỉ số và diễn giải

PSNR/SSIM/MS-SSIM cần GT sạch đã căn chỉnh. Không so ảnh crop/resize lệch với scan
và coi điểm là chất lượng phục hồi. CER/WER cần transcription thật, giữ quy tắc
Unicode/hoa thường/khoảng trắng thống nhất; CER có thể vượt 1 nếu nhiều insertion.
Ảnh không GT chỉ báo OCR nếu có text chuẩn, sharpness và thời gian, kèm đánh giá
đọc thủ công. Sobel sharpness có thể tăng do nhiễu, không đủ kết luận chất lượng.

`ld_map_epe_px` là sai số endpoint bmap theo pixel, không phải Local Distortion
DocUNet chính thức. MS-SSIM hiện dùng Gaussian moments với biên reflect, cần giữ
cùng implementation khi so sánh. Báo cáo rõ package/version khi đối chiếu bài báo.

## 5. Kết quả

Không điền số liệu train/test chưa chạy. `tables/validation_smoke.md` ghi kết quả
kiểm tra chương trình trên ảnh tự sinh sau khi chạy, kèm liên kết bảng CSV và hình.
Thay phần này bằng thực nghiệm trên tập độc lập sau khi train đầy đủ.

| Thí nghiệm | Bảng đầu ra | Tình trạng cần hoàn tất |
|---|---|---|
| Denoise trên cùng nhiễu | denoise_comparison.csv | Chạy trên test sạch đa dạng |
| Biên khi có bóng | edge_comparison.csv | Hiệu chỉnh validation, đánh giá test |
| Pipeline tổng hợp | evaluation.csv | Train đầy đủ và giữ test độc lập |
| Bỏ module/đổi thứ tự | ablation.csv | Thêm checkpoint đã hội tụ |
| Ảnh thật, OCR | real.csv | Thu thập ảnh và text chuẩn |

Thử nghiệm trên 16 vùng chữ in của một ảnh chụp tiếng Anh cho CER 9,15% ở ảnh
gốc, 2,82% với Magic Color, 2,46% với đen trắng mềm và 3,17% với Sauvola.
Kết quả giới hạn ở 284 ký tự trong các vùng cắt tay. Xem
[cấu hình, kết quả và cách tái lập](tables/anh3_printed_ocr.md).

## 6. Hạn chế và hướng phát triển

Không thể khôi phục chữ bị che khuất hoặc mất hoàn toàn do mờ. Baseline dewarp
chỉ mô hình hóa độ cong dọc trên trang nhiều dòng; contour dễ nhầm khi bóng đậm
hoặc nền gần màu giấy. Synthetic 2D có domain gap với trang cong 3D. Mô hình
hợp nhất không cho phép ablation riêng hai tác vụ bằng cách tắt một cờ pipeline.
Pretrained trên ảnh tự nhiên có thể tạo nét giả quanh chữ, cần dùng OCR và kiểm
tra nội dung chứ không chỉ nhìn ảnh. Giới hạn inference 512 px giảm tài nguyên
nhưng có thể mất cải thiện ở chữ nhỏ; đánh giá thêm tiled inference nếu cần.

Ưu tiên tiếp theo: train U-Net trên GPU, tải UVDoc, đánh giá OCR tiếng Việt, sau
đó train hình học và thêm dữ liệu tổ hợp. Lưu checkpoint, seed, manifest và bảng
thử nghiệm để kết quả có thể tái lập.

## Tài liệu tham khảo

- [UVDoc — code, weights và quy ước grid](https://github.com/tanguymagne/UVDoc).
- [DocTr — geometric unwarping và illumination](https://github.com/fh2019ustc/DocTr).
- [NAFNet — inference và pretrained restoration](https://github.com/megvii-research/NAFNet).
- [PyTorch MS-SSIM — loss khả vi](https://github.com/VainF/pytorch-msssim).
