# DỊCH CHI TIẾT PAPER YOLOv6
**"YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications"**
**Tác giả:** Meituan Vision AI Department (美团)
**Năm:** 2022 (v1: 06/2022, v3: 01/2023)
**Link:** https://arxiv.org/abs/2209.02976
**Code:** github.com/meituan/YOLOv6

---

## ABSTRACT

YOLO series đã trở thành tiêu chuẩn cấp công nghiệp cho efficient object detection. Trong technical report này, Meituan đẩy giới hạn YOLO lên mức tiếp theo, tập trung vào **ứng dụng công nghiệp**. Nhóm nghiên cứu tổng hợp các tiến bộ mới nhất từ cả industry và academia: network design, training strategies, testing techniques, quantization, và optimization. YOLOv6-N đạt 35.9% AP ở **1234 FPS** (T4 TensorRT). YOLOv6-S đạt 43.5% AP ở 495 FPS — vượt YOLOv5-S, YOLOX-S, và PPYOLOE-S cùng scale. Phiên bản quantized YOLOv6-S đạt 43.3% AP ở **869 FPS**.

---

## 1. INTRODUCTION

Yêu cầu trong môi trường công nghiệp đa dạng: có lúc cần tốc độ cực cao (camera giám sát realtime), có lúc cần accuracy cao (kiểm tra chất lượng sản phẩm). Meituan phục vụ hàng triệu merchants → cần detector linh hoạt, deploy được trên nhiều loại hardware.

YOLOv6 được thiết kế với triết lý **deployment-ready** — không chỉ accuracy cao trên benchmark mà phải thực sự nhanh khi deploy bằng TensorRT/ONNX.

Đóng góp:
1. Thiết kế lại backbone và neck cho hardware-efficient
2. Cải tiến label assignment và loss function
3. Quantization scheme cho deployment
4. Nhiều model sizes (N/T/S/M/L)

---

## 2. METHOD

### 2.1. Network Design

#### 2.1.1. Backbone: EfficientRep

**Vấn đề:** Multi-branch architectures (ResNet residual, CSP) có accuracy tốt khi training nhưng **chậm khi inference** vì parallel branches không tận dụng hết GPU.

**Giải pháp: Re-parameterization**

Ý tưởng cốt lõi — kiến trúc **training khác inference**:

**Khi Training (multi-branch):**
```
Input → Conv 3×3 ─┐
Input → Conv 1×1 ─┼→ Add → Output
Input → Identity ─┘
```
3 branches song song → model học representations phong phú hơn

**Khi Inference (single branch):**
```
Input → Conv 3×3 → Output
```
Gộp (re-parameterize) 3 branches thành 1 Conv 3×3 duy nhất bằng cách:
- Conv 1×1 pad thành 3×3
- Identity = Conv 3×3 với weights = identity matrix
- Cộng 3 weight matrices → 1 Conv 3×3

Kết quả: Training phong phú như multi-branch, inference nhanh như single Conv.

**RepBlock vs CSPStackRep:**
- Model nhỏ (N/S): Dùng **RepBlock** (stack of RepConv) — đơn giản, nhanh
- Model lớn (M/L): Dùng **CSPStackRep** — CSP structure với RepConv bên trong. Model lớn cần capacity nhiều hơn, CSP giúp giảm computation mà giữ accuracy

#### 2.1.2. Neck: Rep-PAN

PAN (Path Aggregation Network) với RepBlocks thay CSP blocks:
- Top-down: Features từ deep layers → upsample → concat với shallow features
- Bottom-up: Features từ shallow → downsample → concat với deep features
- Mọi convolution block trong PAN đều dùng RepConv → nhanh khi inference

#### 2.1.3. Head: Efficient Decoupled Head

Kế thừa decoupled head từ YOLOX (tách cls/reg) nhưng tối ưu:

**YOLOX head:** 2 branches, mỗi branch có 2 Conv 3×3 → tốn computation

**YOLOv6 head:** 
- Giảm xuống 1 Conv 3×3 + 1 Conv 1×1 mỗi branch
- Thêm **hybrid-channel strategy**: giảm channels trong head → ít computation
- Kết quả: Nhanh hơn YOLOX head mà accuracy tương đương

### 2.2. Anchor-Free Detection

YOLOv6 là **anchor-free** — theo phong cách FCOS:
- Mỗi pixel trên feature map là 1 anchor point
- Dự đoán 4 distances: top, bottom, left, right từ anchor point đến box edges
- Không cần chọn anchor sizes → đơn giản hóa pipeline
- Không cần AutoAnchor → ít hyperparameters

### 2.3. Label Assignment: TAL + SimOTA

**Giai đoạn 1 (early training):** Dùng **SimOTA** (từ YOLOX)
- Optimal Transport Assignment: Tìm matching tối ưu giữa predictions và ground truths
- Xét cả classification cost và localization cost

**Giai đoạn 2 (later training):** Chuyển sang **TAL** (Task-Aligned Learning)
- Alignment metric: t = s^α × u^β
- s = classification score, u = IoU
- α = 1, β = 6 (mặc định)
- Đảm bảo: prediction tự tin → localization tốt

**Tại sao 2 giai đoạn?**
- SimOTA cho early training ổn định hơn (dynamic, linh hoạt)
- TAL cho later training chính xác hơn (aligned metric)
- Kết hợp → tốt hơn dùng 1 trong 2

### 2.4. Loss Functions

**Classification Loss:** VariFocal Loss (VFL)
- Cải tiến Focal Loss: Xử lý class imbalance nhưng **coi positive và negative samples khác nhau**
- Positive samples: Weighted bằng IoU target → high-quality predictions được ưu tiên
- Negative samples: Giảm weight (giống Focal Loss)

**Box Regression Loss:** SIoU / GIoU
- SIoU xét **angle** giữa predicted box và GT → gradient hướng đúng hơn
- Model nhỏ dùng SIoU, model lớn dùng GIoU (thí nghiệm cho thấy khác nhau tùy scale)

**DFL (Distribution Focal Loss):** Dùng cho model M/L
- Dự đoán phân phối thay vì single value (giống v8)
- Chỉ áp dụng cho model lớn (model nhỏ thêm DFL không cải thiện)

---

## 3. QUANTIZATION & DEPLOYMENT

### 3.1. Post-Training Quantization (PTQ)

Chuyển model FP32 → INT8 mà ít mất accuracy:
- **Channel-wise distillation:** Teacher (FP32) hướng dẫn student (INT8) theo từng channel
- Giảm accuracy loss so với naive PTQ

### 3.2. Quantization-Aware Training (QAT)

Training model với fake quantization nodes:
- Model "biết" nó sẽ bị quantize → tự điều chỉnh weights
- **Self-distillation:** Dùng FP32 version của chính model đó làm teacher
- Kết quả: INT8 model gần bằng hoặc bằng FP32 accuracy

### 3.3. RepOptimizer

Vấn đề: RepConv khi quantize bị mất accuracy nhiều (do range khác nhau giữa branches)

Giải pháp: **RepOptimizer** — Tối ưu training process cho quantization-friendly:
- Train với gradient re-parameterization → weights distribution phù hợp quantization hơn
- Kết quả: Quantized model ít bị degradation

---

## 4. EXPERIMENTS

### 4.1. So sánh với SOTA (COCO val2017, T4 TensorRT FP16)

| Model | mAP50-95 | FPS (T4 FP16) | Params |
|-------|---------|:-------------:|--------|
| YOLOv5-N | 28.0% | 602 | 1.9M |
| **YOLOv6-N** | **37.5%** | **1187** | **4.7M** |
| YOLOX-S | 40.5% | 333 | 9.0M |
| YOLOv5-S | 37.4% | 349 | 7.2M |
| **YOLOv6-S** | **45.0%** | **495** | **18.5M** |
| YOLOv5-M | 45.4% | 176 | 21.2M |
| **YOLOv6-M** | **50.0%** | **233** | **34.9M** |
| YOLOv5-L | 49.0% | 113 | 46.5M |
| **YOLOv6-L** | **52.8%** | **116** | **59.6M** |

**Nhận xét:** YOLOv6 vượt YOLOv5 ở MỌI scale về cả accuracy và speed.

### 4.2. Quantization Results

| Model | FP16 mAP | INT8 mAP | FP16 FPS | INT8 FPS |
|-------|---------|---------|---------|---------|
| YOLOv6-S | 43.5% | 43.3% | 495 | **869** |

INT8 chỉ mất **0.2% mAP** nhưng tăng **75% FPS** — deployment rất hiệu quả.

### 4.3. Ablation Study

**Backbone:**

| Backbone | mAP | FPS |
|----------|-----|-----|
| CSPDarknet (v5) | 35.4% | 337 |
| **EfficientRep** | **37.5%** | **525** | 

EfficientRep: +2.1% mAP VÀ +56% FPS → re-param hiệu quả.

**Head:**

| Head | mAP | Latency |
|------|-----|---------|
| Coupled (v5-style) | 36.1% | 0.83ms |
| Decoupled (YOLOX) | 36.4% | 0.93ms |
| **Efficient Decoupled** | **37.5%** | **0.81ms** |

Efficient Decoupled: Accuracy cao nhất VÀ nhanh nhất.

**Label Assignment:**

| Strategy | mAP |
|----------|-----|
| SimOTA only | 36.6% |
| TAL only | 36.9% |
| **SimOTA → TAL** | **37.5%** |

2-stage assignment tốt nhất.

---

## 5. CONCLUSION

YOLOv6 tập trung vào industrial deployment — không chỉ accuracy mà còn quantization, export, và inference speed thực tế. Re-parameterization là kỹ thuật chủ đạo: train rich, deploy fast. Quantization pipeline (PTQ + QAT) cho phép INT8 inference gần bằng FP32. Meituan dùng YOLOv6 trong production phục vụ hàng triệu merchants.

---

## TÓM TẮT KỸ THUẬT MỚI CỦA YOLOv6

| Kỹ thuật | Mô tả | Ảnh hưởng |
|---------|-------|----------|
| **EfficientRep** | Re-param backbone | +2% mAP, +56% FPS |
| **Rep-PAN** | PAN với RepConv | Faster neck |
| **Efficient Decoupled Head** | Lightweight decoupled | Nhanh hơn YOLOX head |
| **Anchor-Free** | FCOS-style | Đơn giản hóa |
| **SimOTA → TAL** | 2-stage assignment | +0.9% mAP |
| **VFL + SIoU** | Loss functions | Cải thiện training |
| **PTQ + QAT** | Quantization | INT8 ≈ FP32, +75% FPS |
| **RepOptimizer** | Quant-friendly training | Giảm quantization loss |

---

*Bản dịch chi tiết từ paper gốc arXiv:2209.02976 (Meituan Technical Report).*
