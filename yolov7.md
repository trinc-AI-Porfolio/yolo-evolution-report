# DỊCH CHI TIẾT PAPER YOLOv7
**"YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors"**
**Tác giả:** Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao
**Hội nghị:** CVPR 2023 | **Link:** https://arxiv.org/abs/2207.02696

---

## ABSTRACT

YOLOv7 vượt trội tất cả object detectors (cả YOLO-based và non-YOLO) về tốc độ và accuracy trong dải 5-160 FPS. YOLOv7-E6 (56 FPS, V100) đạt 55.9% mAP — vượt transformer-based SWINL Cascade-Mask R-CNN (53.9%) và convolutional-based ConvNeXt-XL Cascade-Mask R-CNN (55.2%). YOLOv7 chỉ dùng COCO dataset, không pretrain models khác.

---

## 1. INTRODUCTION

Real-time object detection là thành phần quan trọng trong nhiều hệ thống CV. Phần cứng computing ngày càng rẻ → nhu cầu real-time detector accuracy cao tăng. Real-time detectors phải tối ưu cho mobile GPU, GPU, hoặc general GPU.

Bài toán chính: Làm sao tăng accuracy mà KHÔNG tăng inference cost? → "Trainable Bag-of-Freebies" — kỹ thuật chỉ tăng training cost.

Các vấn đề nghiên cứu:
1. Kiến trúc re-parameterization tương tác thế nào với các chiến lược mạng khác nhau?
2. Chiến lược dynamic label assignment ảnh hưởng thế nào đến training với auxiliary head?

---

## 2. RELATED WORK

### 2.1. Real-time Object Detectors
Hiện tại SOTA real-time detectors dựa trên YOLO và FCOS. Để trở thành SOTA cần: (1) Backbone hiệu quả hơn, (2) Path-aggregation neck hiệu quả hơn, (3) Thiết kế detection head hiệu quả hơn.

### 2.2. Model Re-parameterization
Chia feature-level ensemble và module-level ensemble. RepConv, RepVGG gộp multi-branch thành single conv khi inference.

### 2.3. Model Scaling
Scale depth/width/resolution đồng thời (EfficientNet compound scaling). Nhưng cho concat-based models cần cách khác.

---

## 3. ARCHITECTURE

### 3.1. Extended Efficient Layer Aggregation Network (E-ELAN)

**ELAN** (Efficient Layer Aggregation Network): Thiết kế dựa trên phân tích gradient path. Sử dụng expand, shuffle, merge cardinality để học nhiều features hơn mà không phá gradient path gốc.

**E-ELAN** mở rộng ELAN bằng cách:
- Expand: Thêm cardinality (nhiều nhánh tính toán song song)
- Shuffle: Xáo trộn features từ các nhánh
- Merge: Gộp lại → diverse feature representation

Quan trọng: Không thay đổi gradient path ban đầu → giữ stability khi scale model lên.

### 3.2. Model Scaling cho Concatenation-Based Models

Khi scale depth (thêm layers) trong concat-based architecture → output width tăng theo. Phải scale depth VÀ width đồng thời sao cho input/output ratio giữ nguyên khi concat.

Công thức: Khi scale depth factor d, phải scale transition layer width lên w×d để giữ cân bằng.

---

## 4. TRAINABLE BAG-OF-FREEBIES

### 4.1. Planned Re-parameterized Convolution

**Phát hiện quan trọng:** RepConv (re-parameterized convolution) KHÔNG nên dùng khi có **identity connection** (residual shortcut).

Giải thích: RepConv có 3 branches (3×3, 1×1, identity). Khi gộp vào residual block (đã có identity), identity bị lặp → phá vỡ đặc tính residual → accuracy giảm.

**Giải pháp: RepConvN** — RepConv KHÔNG có identity branch. Dùng khi kết hợp với residual connections.

Thí nghiệm chứng minh:
- RepConv trong PlainNet: OK ✅
- RepConv trong ResNet: Giảm accuracy ❌
- **RepConvN** trong ResNet: OK ✅

### 4.2. Coarse-to-Fine Lead Head Guided Label Assignment

**Vấn đề:** Khi dùng auxiliary head (head phụ để hỗ trợ training), label assignment như thế nào?

**Giải pháp: Lead head guided:**
- **Lead head:** Head chính (dùng khi inference) → fine-grained label assignment (chặt, chính xác)
- **Auxiliary head:** Head phụ (chỉ dùng khi training, bỏ khi inference) → coarse-grained label assignment (lỏng hơn, chấp nhận nhiều positives)

**Coarse-to-fine:** 
1. Dùng lead head predictions tạo soft labels
2. Auxiliary head học từ soft labels này (coarse)
3. Lead head học fine-grained

Ưu điểm: Auxiliary head giúp tạo gradient phong phú hơn → model học representations tốt hơn → lead head chính xác hơn. Bỏ auxiliary head khi inference → KHÔNG tăng cost.

### 4.3. Các BoF khác

- **Batch normalization** trong conv-bn-activation
- **Implicit knowledge** giống YOLOR — thêm implicit representations
- **EMA (Exponential Moving Average)** model cho inference

---

## 5. EXPERIMENTS

### 5.1. Experimental Setup
Train trên COCO train2017 (118k images). Evaluate trên val2017. Test trên test-dev2017.
Input resolution: 640 (base models), 1280 (large models).
Không dùng pretrained backbone ngoài ImageNet (khác nhiều papers khác).

### 5.2. So sánh với SOTA

**Base models (khoảng 30-80 FPS):**

| Model | Params | FPS (V100) | mAP50-95 |
|-------|--------|-----------|---------|
| YOLOv5-N | 1.9M | 159 | 28.0% |
| YOLOv5-L | 46.5M | 99 | 48.8% |
| YOLOX-L | 54.2M | 68 | 50.0% |
| PPYOLOE-L | 52.2M | 78 | 51.4% |
| **YOLOv7** | **36.9M** | **161** | **51.4%** |
| **YOLOv7-X** | **71.3M** | **114** | **53.1%** |

**Large models (khoảng 15-56 FPS):**

| Model | FPS (V100) | mAP50-95 |
|-------|-----------|---------|
| YOLOv7-W6 | 84 | 54.9% |
| YOLOv7-E6 | 56 | 55.9% |
| **YOLOv7-E6E** | **36** | **56.8%** |
| SWIN-L Cascade R-CNN | 12 | 53.9% |
| ConvNeXt-XL Cascade R-CNN | - | 55.2% |

**Kết luận:** YOLOv7-E6E (56.8% mAP) vượt tất cả detectors tại thời điểm đó — cả transformer-based lẫn convolution-based.

### 5.3. Ablation Study

**E-ELAN vs ELAN:**
ELAN → E-ELAN: +0.7% mAP mà không tăng params/FLOPs đáng kể.

**Planned Re-parameterized Conv:**
RepConv trong model có residual: -0.2% mAP
RepConvN trong model có residual: +0.3% mAP → planned re-param hoạt động.

**Auxiliary Head:**
Không auxiliary: baseline
Lead head guided coarse: +0.4% mAP
Lead head guided fine: +0.2% mAP (ít hơn coarse)
Coarse-to-fine: **+0.5% mAP** (tốt nhất)

---

## 6. CONCLUSION

YOLOv7 đề xuất kiến trúc mới (E-ELAN) và "trainable bag-of-freebies" (planned re-param, coarse-to-fine label assignment). Đạt SOTA accuracy và speed, vượt tất cả detectors trong dải 5-160 FPS mà chỉ train trên COCO.

---

*Bản dịch chi tiết từ paper gốc arXiv:2207.02696.*
