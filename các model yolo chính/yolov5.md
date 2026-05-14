# YOLOv5 — TÀI LIỆU KỸ THUẬT CHI TIẾT (Không có paper chính thức)
**Tác giả:** Glenn Jocher / Ultralytics
**Năm:** 2020 (release 06/2020, cập nhật liên tục đến 2023)
**Nguồn:** GitHub (github.com/ultralytics/yolov5), Ultralytics Docs
**Framework:** PyTorch (chuyển từ Darknet C của v1-v4)

---

## BỐI CẢNH

YOLOv5 release chỉ **2 tháng** sau YOLOv4 (04/2020 vs 06/2020), gây tranh cãi lớn:
- Không có paper, không peer review
- Tên "v5" bị cộng đồng phản đối (nhiều người cho rằng không xứng đáng kế thừa tên YOLO)
- Tuy nhiên, v5 nhanh chóng trở thành **YOLO phổ biến nhất** nhờ ecosystem vượt trội

**Nguồn thông tin:** Toàn bộ thông số dưới đây từ GitHub README, Ultralytics docs, release notes, và community benchmarks. Không có peer-reviewed publication.

---

## KIẾN TRÚC

### Backbone: CSPDarknet53 (Modified)
- Kế thừa từ YOLOv4 nhưng tối ưu cho PyTorch
- **Focus layer** (v5 early): Slice ảnh thành 4 phần (mỗi phần lấy pixel cách 1) → concat → giảm spatial 2× nhưng tăng channels 4× → giảm computation ở đầu pipeline. Sau đó thay bằng Conv 6×6 stride 2 ở versions sau.
- CSP Bottleneck (**C3 module**): Cross Stage Partial với 3 convolutions. Chia feature thành 2 phần, 1 phần qua bottleneck stack, 1 phần bypass → concat

### Neck: PANet (modified) + SPPF
- **SPPF (Spatial Pyramid Pooling - Fast):** Thay vì 3 parallel MaxPool (5, 9, 13) như SPP gốc → SPPF dùng 3 sequential MaxPool 5×5. Kết quả tương đương nhưng **nhanh hơn 2×**
- PANet: Top-down + bottom-up feature aggregation

### Head: Anchor-based, 3 scales
- 3 detection scales: P3 (80×80), P4 (40×40), P5 (20×20) tại input 640
- 3 anchors/scale = 9 anchors tổng
- **AutoAnchor:** Tự động tính toán anchor tối ưu cho dataset bằng k-means + genetic algorithm. Nếu best possible recall < 98% với anchors mặc định → tự tính lại

### 5 Model Sizes

| Model | Params | FLOPs | mAP50-95 | Latency (V100) |
|-------|--------|-------|---------|---------------|
| v5n (Nano) | 1.9M | 4.5G | 28.0% | 0.7ms |
| v5s (Small) | 7.2M | 16.5G | 37.4% | 1.0ms |
| v5m (Medium) | 21.2M | 49.0G | 45.4% | 2.5ms |
| v5l (Large) | 46.5M | 109.1G | 49.0% | 3.5ms |
| v5x (XLarge) | 86.7M | 205.7G | 50.7% | 5.0ms |

*Nguồn: Ultralytics GitHub README, đo trên COCO val2017, input 640×640, batch=1, V100.*

---

## PHƯƠNG PHÁP TRAINING

### Data Augmentation
- **Mosaic:** Ghép 4 ảnh (kế thừa v4)
- **Copy-Paste augmentation:** Copy object từ ảnh này paste sang ảnh khác
- **Random affine:** Scale (±50%), translate, rotate
- **HSV augmentation:** Hue ±1.5%, Saturation ±70%, Value ±40%
- **Random flip:** Horizontal 50%
- **MixUp:** Trộn 2 ảnh (probability thấp)

### Loss Function
- **Box loss:** CIoU Loss (kế thừa v4)
- **Objectness loss:** BCE (Binary Cross Entropy)
- **Classification loss:** BCE
- Objectness loss được **scale theo grid size** — scale nhỏ (vật lớn) weight thấp hơn, scale lớn (vật nhỏ) weight cao hơn

### Training Config
- Optimizer: SGD (momentum=0.937, weight_decay=0.0005)
- Learning rate: Linear warmup (3 epochs) → Cosine annealing → lr_final = lr_init × 0.01
- 300 epochs mặc định
- Label smoothing: 0.0 (mặc định)
- **Hyperparameter Evolution:** Genetic algorithm chạy 300+ thế hệ tìm hyperparams tối ưu

### Export & Deploy
- Hỗ trợ 10+ formats: PyTorch, ONNX, TensorRT, CoreML, TFLite, OpenVINO, PaddlePaddle, Edge TPU...
- **TensorRT FP16** cho production inference
- Ultralytics HUB cho cloud training/deploy

---

## ĐÓNG GÓP CHÍNH

1. **Dân chủ hóa YOLO:** `pip install` thay vì compile C code → hạ rào cản entry
2. **AutoAnchor:** Tự tìm anchors cho dataset cụ thể
3. **SPPF:** Tăng tốc SPP
4. **5 model sizes:** Linh hoạt từ mobile đến server
5. **Export pipeline:** 10+ formats, production-ready
6. **Hyperparameter evolution:** Tìm config tối ưu tự động

## HẠN CHẾ
1. Không có paper → không có peer review, khó kiểm chứng claims
2. Kiến trúc gần giống v4, đóng góp kỹ thuật mới hạn chế
3. Vẫn anchor-based → cần AutoAnchor, phức tạp hơn anchor-free
4. Naming gây tranh cãi trong cộng đồng

---

*Tổng hợp từ: Ultralytics GitHub, Docs, Release Notes. Không phải peer-reviewed paper.*
