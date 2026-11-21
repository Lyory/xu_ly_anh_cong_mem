# Doc Restore — Khôi phục ảnh tài liệu cong, bóng đổ và mờ

Project cuối kỳ xử lý ảnh, gồm baseline OpenCV, U-Net attention tự train,
DewarpNet ResNet18 tự train, adapter pretrained, OCR và demo Gradio.
Thư mục hiện tại là gốc project, không cần tạo thêm thư mục `doc-restore`.

## Quét một ảnh bằng mô hình đã huấn luyện

Clone dự án, cài môi trường theo mục **Cài đặt**, rồi dùng ảnh của bạn (thay đường dẫn trong ví dụ):

```bash
python scan.py duong/dan/tai-lieu.jpg --style magic-color --output outputs/tai-lieu.png --pdf
```

Lệnh ghi PNG và PDF cùng tên. Bỏ `--pdf` để chỉ xuất ảnh cùng các bước trung gian. Có thể chọn `soft-bw` hoặc `sauvola` cho ảnh xám hoặc ảnh nhị phân. Chế độ scan chỉ cần ảnh đầu vào. Pipeline dùng UVDoc khi có trọng số cục bộ và phát hiện bố cục phù hợp. Thư mục `data/real_test` dùng để kiểm thử; dữ liệu huấn luyện chỉ cần cho các thí nghiệm huấn luyện riêng.

| Tệp | Chức năng trong luồng quét |
| --- | --- |
| `scan.py`, `scripts/restore.py` | Chọn chế độ scan mặc định, đọc tham số và ghi kết quả. |
| `src/pipeline.py` | Điều phối từ đọc ảnh đến áp dụng kiểu xuất. |
| `src/classical/paper_segmentation.py` | Tìm tờ giấy ở tâm và loại phần giấy khác dính vào mép. |
| `src/classical/scan.py` | Nắn theo đường bao, phối hợp UVDoc với đường kẻ và bù sáng từng kênh. |
| `src/classical/form_dewarp.py` | Nội suy lưới lấy mẫu theo các cạnh biểu mẫu và hàng bảng. |
| `src/classical/binarize.py` | Áp dụng đường cong xám hoặc ngưỡng Sauvola và làm sạch rìa. |
| `src/models/pretrained_wrappers.py`, `src/ocr.py` | Gọi mô hình pretrained và xuất văn bản/PDF bằng Tesseract. |

## Cài đặt

Python 3.10 trở lên; đã kiểm tra bằng Python 3.12 trên CPU. Tại thư mục gốc:

```bash
git clone https://github.com/Lyory/xu_ly_anh_cong_mem.git
cd xu_ly_anh_cong_mem
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU; nếu dùng GPU, cài cặp torch/torchvision tương thích CUDA trước.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

OCR cần executable hệ thống, không chỉ package Python:

Project dùng bản `opencv-python-headless` để chạy được trên máy chủ/Kaggle không
cần thư viện cửa sổ; các thuật toán OpenCV giống bản có GUI.

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-vie
tesseract --list-langs
```

Không cần Tesseract để chạy phục hồi ảnh, sinh dữ liệu hoặc train. RemBG/VietOCR,
BEDSR-Net và DeblurGAN-v2 là hướng mở rộng, chưa tích hợp. Không tự tải dataset,
checkpoint hoặc VGG16; VGG16 chỉ tải khi bật perceptual loss.

Ảnh cá nhân trong `data/real_test/` được giữ trên máy người dùng. Bạn có thể tải ảnh của mình trực tiếp trong demo.

## Chạy ngay

```bash
source .venv/bin/activate
python -m app.demo
```

Mở `http://127.0.0.1:7860`. Chọn **Quét tài liệu màu** để tự tách nền, nắn theo
đường bao giấy và làm sáng nền, giữ màu chữ. Chế độ này dùng ảnh chụp một tờ
giấy sáng chiếm phần lớn khung hình. Bật **Nhận dạng chữ và tạo PDF** nếu cần.
Chế độ học sâu cần checkpoint đã train.
CLI nhận đường dẫn ảnh, xử lý EXIF và xuất ảnh, ảnh trung gian, metadata JSON:

```bash
python -m scripts.restore data/real_test/page.jpg --output outputs/page.png
python -m scripts.restore data/real_test/page.jpg --binarize --ocr --pdf --lang vie+eng
python -m scripts.restore data/real_test/page.jpg --mode scan --output outputs/scan/page.png --ocr --pdf
```

Mọi lệnh Python trong tài liệu chạy bằng `python -m ...` từ gốc project.
Trong Gradio, chọn **Kiểu ảnh xuất**. Dùng **Giữ màu (Magic Color)** để giữ màu
mực, **Đen trắng mềm** để giữ sắc xám của nét mảnh, hoặc **Nhị phân Sauvola**
để xuất ảnh chỉ còn hai mức đen và trắng. Nếp gấp sâu có thể còn dấu và vùng
giấy nằm ngoài khung ảnh sẽ thiếu nội dung trong kết quả.

```bash
python -m scripts.restore data/real_test/anh3.jpg --mode magic-color --output outputs/scan/anh3_magic-color.png
python -m scripts.restore data/real_test/anh3.jpg --mode soft-bw --output outputs/scan/anh3_soft-bw.png
python -m scripts.restore data/real_test/anh3.jpg --mode sauvola --output outputs/scan/anh3_sauvola.png
```

`--mode scan --style magic-color|soft-bw|sauvola` cũng chọn được kiểu ảnh.
Với `--mode scan`, cờ cũ `--binarize` vẫn chọn Sauvola. Bản Sauvola dùng cửa
sổ theo kích thước trang và hạn chế ngưỡng ở vùng giấy sáng. Chế độ quét
lọc nét nhỏ sát biên sau khi nắn và giữ các đường kẻ dài.

Kết quả CER trên chữ in của `anh3.jpg` và cách chạy lại được ghi trong
[bảng CER](report/tables/anh3_printed_ocr.md).

Chế độ `scan` ưu tiên tờ giấy ở giữa ảnh. Nếu có khung biểu mẫu và các đường
kẻ dài ở trên và dưới, nó dùng các đường đó để nắn nội dung và cắt lề sát hơn.
Với hóa đơn có bảng nhỏ, chế độ này nắn các hàng in đã xác định được và giữ
toàn trang, bao gồm tiêu đề, chú thích và con dấu bên dưới bảng.
Nếu có `external/UVDoc/model.py` và `external/UVDoc/model/best_model.pkl`, bước
nắn biểu mẫu thử thêm UVDoc và kiểm tra lại các đường kẻ sau khi nắn. Bản clone đã kèm trọng số UVDoc cho bước này. [UVDoc và trọng số chính thức](https://github.com/tanguymagne/UVDoc/tree/4c9b82b537057aff2526e6dd118a847cdd072e82).

## Pipeline

```text
EXIF → resize giữ tỉ lệ / cân bằng trắng tùy chọn → chẩn đoán
→ biên Canny/Sobel/... → contour 4 góc → phối cảnh, cắt nền
→ deskew Hough / projection / minAreaRect
→ dewarp cổ điển / DewarpNet / UVDoc / DocTr
→ khử bóng → khử nhiễu → khử mờ
→ CLAHE / Sauvola tùy chọn → OCR / searchable PDF tùy chọn
```

- `src/classical/preprocess.py`: EXIF, RGB, resize, gray-world, padding.
- `edge_detection.py`: Sobel, Prewitt, Scharr, Laplacian, LoG, Canny; dùng thật
  trong phát hiện tài liệu, Hough, chẩn đoán và sharpness.
- `denoise.py`: mean, Gaussian, median, bilateral, NLM, Wiener không gian;
  khử nhiễu trước deconvolution và median nhẹ trước Sauvola.
- `deskew.py`: ba cách ước lượng góc; giới hạn ±12°, canvas giữ nguyên có thể cắt góc.
- `frequency.py`: phổ FFT có cửa sổ Hann, homomorphic, PSF motion heuristic.
- `dewarp_classic.py`: nối dòng chữ, fit đa thức bậc 2, nội suy độ cong dọc;
  cần ít nhất ba dòng đáng tin cậy, không phù hợp ảnh ít chữ/bảng phức tạp.
- `shadow_removal.py`: chia nền dilate/median hoặc homomorphic.
- `deblur.py`: unsharp, Wiener trong miền tần số, Richardson–Lucy.

API thống nhất **RGB uint8 H×W×3**; tensor RGB float trong [0,1]. `restore()` trả
dict `image`, `intermediates`, `metadata`, `bmap`. Không tìm được contour thì giữ
ảnh và ghi cảnh báo. Điểm chẩn đoán là heuristic; `blur_score` là phương sai
Laplacian, càng cao thường càng nét; cần hiệu chỉnh ngưỡng trên tập validation.

```python
from src.pipeline import DocRestorer
restorer = DocRestorer(
    mode="classical", edge_method="canny", deskew_method="projection",
    denoise_method="bilateral", deblur_method="wiener", estimate_psf=True,
    white_balance=True, modules={"contrast": True, "binarize": False})
result = restorer.restore("data/real_test/page.jpg")
```

PSF từ một ảnh là bài toán thiếu xác định: bộ ước lượng FFT chỉ gợi ý hướng,
độ dài mặc định 7 px do người dùng cung cấp. Không dùng như phép đo chính xác.

## Dữ liệu và backward map

Đặt ảnh sạch được phép sử dụng trong `data/raw_clean/`; ảnh thật trong
`data/real_test/`; nền trong `data/backgrounds/`. Nên dùng tối thiểu vài trăm tài
liệu sạch độc lập và 100–150 ảnh chụp thật, gồm tiếng Việt, bảng và nhiều thiết bị.

```bash
python -m scripts.render_pdf /path/to/pdfs --output data/raw_clean --dpi 150
python -m scripts.make_dataset --variants 8 --size 512 --seed 42
python -m scripts.make_warp_dataset --variants 8 --size 512 --seed 42
```

Chia theo tài liệu nguồn khoảng 80/10/10 **trước** khi sinh biến thể. Tên trang PDF
có dạng `document_hash__p0001.png` để các trang cùng PDF nằm trong một split.
Ảnh tự đưa vào cần dùng cùng quy ước `group__pNNNN` nếu cùng tài liệu. Với hai PDF
nội dung trùng nhưng khác tên cần tự loại trùng trước. Cần ít nhất ba nhóm.
Letterbox lên canvas vuông, không kéo méo tỉ lệ. `manifest.json` lưu nguồn, seed,
split. Generator yêu cầu thư mục output mới để tránh lẫn dữ liệu cũ.

`warp/.../bmap/*.npy`: float32 **H×W×2**, tọa độ tuyệt đối chuẩn hóa [-1,1], thứ tự
(x,y), `align_corners=True`. Tại một pixel ảnh sạch, bmap chỉ pixel cần lấy từ ảnh
cong. DewarpNet dự đoán **flow = bmap − identity_grid**, B×2×H×W; flow cũng dùng
đơn vị chuẩn hóa, không phải pixel. Khi resize bmap chuẩn hóa không nhân lại flow.
Generator dùng biến dạng trơn sin/cuộn và perspective, có nghịch đảo lặp để giữ
đúng ground truth. Đây là mô phỏng 2D nhẹ, chưa mô phỏng occlusion/nếp gấp tự che
khuất hoặc hình học 3D của giấy thật.

`paste_on_background` là tiện ích cho thí nghiệm contour; chưa ghép vào dataset
train để tránh thay đổi hệ tọa độ mà không cập nhật bmap. Nhiễu tổng hợp gồm
Gaussian, JPEG, lệch trắng; tham số `salt_pepper` bổ sung muối tiêu.

## Huấn luyện

```bash
python -m scripts.train_shadow_deblur --config configs/shadow_deblur.yaml
python -m scripts.train_dewarp --config configs/dewarp.yaml
tensorboard --logdir experiments/checkpoints
```

- U-Net: bốn tầng, attention gate, residual RGB; hỗ trợ 3 kênh hoặc kênh nền thứ 4.
- DewarpNet: encoder ResNet18, decoder skip, 2 kênh flow, `grid_sample` khả vi.
- `CombinedLoss`: L1 + SSIM + VGG16 tùy chọn + Sobel gradient magnitude.
- `DewarpLoss`: map L1 + ảnh unwarp L1/SSIM + regularization gradient flow.
- Optimizer AdamW, cosine LR, CUDA AMP, gradient clipping, seed và TensorBoard.
- Mỗi run lưu `best.pt`, `last.pt`, config và log epoch tại `experiments/log.csv`.

Resume với cùng kiến trúc và lịch epochs; `epochs` là tổng số epoch:

```bash
python -m scripts.train_shadow_deblur --resume experiments/checkpoints/shadow_deblur/last.pt
python -m app.demo --shadow-checkpoint experiments/checkpoints/shadow_deblur/best.pt
python -m scripts.restore data/real_test/page.jpg --mode hybrid \
  --shadow-checkpoint experiments/checkpoints/shadow_deblur/best.pt \
  --dewarp-checkpoint experiments/checkpoints/dewarp/best.pt
```

U-Net chung bóng/mờ chạy một lần tại stage `restore_dl` sau khử nhiễu; stage
`shadow`/`deblur` cổ điển tắt khi dùng nó. NAFNet chỉ khử mờ, nên hybrid vẫn giữ
khử bóng cổ điển. Vì vậy ablation mô hình hợp nhất bỏ
`restore_dl` như một khối, không giả định có thể tách hai nhiệm vụ trong weight
chung. `dl` chỉ chạy các tác vụ có weight; `hybrid` dùng dewarp cổ điển nếu chưa có
weight hình học. Cả hai vẫn có thể dùng tiền xử lý/khử nhiễu cổ điển theo cấu hình.
Nhánh neural dự đoán ở tối đa 512 px và nội suy flow/residual về kích thước gốc;
đây là đánh đổi tài nguyên, có thể làm giảm chất lượng khử mờ chữ nhỏ.

## Pretrained đối chứng

Tham khảo repo chính thức: [UVDoc](https://github.com/tanguymagne/UVDoc),
[DocTr](https://github.com/fh2019ustc/DocTr),
[NAFNet](https://github.com/megvii-research/NAFNet).
Tải weight theo hướng dẫn và giấy phép từng repo; không có weight kèm project.

UVDoc có adapter trực tiếp, đúng input RGB 488×712, grid normalized và
`align_corners=True` theo [mã inference chính thức](https://github.com/tanguymagne/UVDoc/blob/main/demo.py).
Sao chép `configs/pretrained.example.yaml`, điền đường dẫn rồi:

```bash
python -m scripts.restore data/real_test/page.jpg --mode hybrid \
  --backend-config configs/pretrained.local.yaml
python -m app.demo --backend-config configs/pretrained.local.yaml
```

NAFNet và DocTr gọi CLI chính thức bằng subprocess, trong môi trường riêng,
do dependency legacy. NAFNet cần config khớp width và weight. DocTr gốc cần CUDA,
weight segmentation/GeoTr và cả IllTr mặc định vì upstream khởi tạo nó kể cả khi
chỉ chạy hình học. Các adapter này chưa được kiểm chứng bằng pretrained weight
trên máy hiện tại; lỗi upstream được chuyển tiếp, không tự thay bằng model ngẫu nhiên.

## Đánh giá và ablation

```bash
python -m scripts.evaluate --input data/synthetic/shadow_deblur/test/input \
  --target data/synthetic/shadow_deblur/test/target --save-images
python -m scripts.evaluate --input data/synthetic/warp/test/input \
  --target data/synthetic/warp/test/target --bmap data/synthetic/warp/test/bmap \
  --dewarp-checkpoint experiments/checkpoints/dewarp/best.pt \
  --output report/tables/warp.csv
python -m scripts.run_ablation --input data/synthetic/shadow_deblur/test/input \
  --target data/synthetic/shadow_deblur/test/target
python -m scripts.evaluate --input data/real_test --text data/real_text \
  --ocr --lang vie+eng --output report/tables/real.csv
python -m scripts.compare_filters --clean data/raw_clean --limit 10
```

`--text` chứa `.txt` cùng tên/tương đối với ảnh. Paired target/bmap cũng giữ tên
và cây con giống input. Mặc định đánh giá input và classical; truyền checkpoint
để thêm dl/hybrid, hoặc `--configs configs/evaluation.example.yaml`.
Lệnh tạo CSV chi tiết, CSV trung bình và JSON cấu hình để tái lập.

PSNR/SSIM/MS-SSIM chỉ hợp lệ trên ảnh căn chỉnh cùng kích thước; không resize GT
để ép điểm. Đánh giá paired tắt perspective/deskew để giữ hệ tọa độ. Ảnh thật
không có ảnh sạch thì để trống ba chỉ số này. CER/WER chỉ tính khi bật OCR và có
text chuẩn. Sharpness Sobel chịu ảnh hưởng của nhiễu/contrast, không thay PSNR.

**Local Distortion:** `ld_map_epe_px` là endpoint error theo pixel giữa hai bmap,
một **proxy hình học** có GT, không phải LD DocUNet chính thức (SIFT-flow/MATLAB).
Không có map GT thì để trống. MS-SSIM trong `src/metrics.py` dùng Gaussian moments,
reflect border và giảm số thang cho ảnh nhỏ; có thể lệch nhẹ so với implementation
benchmark valid-window. CER/WER mặc định giữ nguyên hoa/thường/dấu; cần thống nhất
quy tắc normalization và psm trước khi báo cáo.

## Kiểm thử và dữ liệu nhỏ

```bash
python -m pytest -q
python -m scripts.make_demo_data --output experiments/smoke/clean
python -m scripts.make_dataset --clean experiments/smoke/clean \
  --output experiments/smoke/shadow --size 128 --variants 1
python -m scripts.make_warp_dataset --clean experiments/smoke/clean \
  --output experiments/smoke/warp --size 128 --variants 1
```

Để kiểm tra một batch train, sửa `data_root`, `output_dir`, `image_size=64`,
`batch_size=2`, `num_workers=0`, U-Net `base=4` trong bản sao config, rồi chạy
`--epochs 1 --max-batches 1`. Checkpoint có nhãn `smoke_only=true`; không dùng kết
quả đó để tuyên bố chất lượng. Tài liệu tự sinh có nội dung rất giống nhau, chỉ
phục vụ kiểm tra chương trình. Cần train đầy đủ trên tài liệu đa dạng để bảo vệ đồ án.

## Notebook và báo cáo

```bash
python -m pip install -r requirements-notebooks.txt
python -m ipykernel install --user --name doc-restore --display-name "Python (doc-restore)"
```

Chọn kernel môi trường này trong Jupyter/VS Code. Trên Kaggle có thể dùng kernel
có sẵn và cài các dependency còn thiếu.

- `01_classical_exploration.ipynb`: ảnh/biên, deskew, FFT và PSF.
- `02_train_kaggle.ipynb`: đường dẫn Kaggle/GPU, sinh dữ liệu, train và lưu weight.
- `03_results_analysis.ipynb`: tổng hợp CSV, ablation, biểu đồ và giới hạn đánh giá.
- `04_filter_comparison.ipynb`: cùng dữ liệu nhiễu, bảng PSNR/SSIM; độ ổn định contour
  khi có bóng, so sánh sáu bộ lọc biên.
- `report/REPORT.md`: lý thuyết, phương pháp, quy trình thực nghiệm và giới hạn.

Lịch 5 tuần: (1) dữ liệu/baseline; (2) biên, nhiễu, deskew, đánh giá cổ điển;
(3) U-Net; (4) dewarp DL/pretrained; (5) ảnh thật, ablation, demo và báo cáo.
