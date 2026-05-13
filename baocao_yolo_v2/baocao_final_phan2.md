# Chương 4: THẾ HỆ GIỮA — YOLOv5–v8 (2020–2023)

## 4.1. YOLOv5 (2020) — Dân chủ hóa YOLO

**Tác giả:** Glenn Jocher / Ultralytics | **Framework:** PyTorch | *Không có paper chính thức* [5]

### 4.1.1. Ý nghĩa lịch sử

Chuyển từ Darknet (C) sang PyTorch. Gây tranh cãi vì không có paper, nhưng trở thành **YOLO phổ biến nhất** nhờ `pip install ultralytics`.

> **Ẩn dụ:** Trước v5, dùng YOLO giống phải **lái xe số sàn** (compile C code, cấu hình .cfg). v5 chuyển sang **xe số tự động** — 3 dòng Python là chạy.

### 4.1.2. Kiến trúc

- **Backbone:** CSPDarknet53 (modified) với **C3 module** — CSP Bottleneck 3 convolutions
- **Neck:** PANet + **SPPF** (SPP Fast) — 3 MaxPool $5 \times 5$ nối tiếp thay vì 3 MaxPool song song → nhanh hơn ~2× trên GPU
- **Head:** Anchor-based, 3 scales, **AutoAnchor** (k-means + genetic algorithm tự tìm anchors tối ưu)

> **Ẩn dụ SPPF:** SPP giống 3 người **cùng lúc** đọc 3 cuốn sách → cần 3 bàn. SPPF giống 1 người đọc **nối tiếp** 3 cuốn → chỉ cần 1 bàn, kết quả tương đương.

### 4.1.3. Kết quả

| Model | Params | FLOPs | mAP@50:95 | Phù hợp |
|:--|:-:|:-:|:-:|:--|
| v5n | 1.9M | 4.5G | 28.0% | Mobile, IoT |
| v5s | 7.2M | 16.5G | 37.4% | Edge devices |
| v5m | 21.2M | 49.0G | 45.4% | Cân bằng |
| v5l | 46.5M | 109.1G | 49.0% | Server |
| v5x | 86.7M | 205.7G | 50.7% | Max accuracy |

*Ghi chú: Đo trên COCO val2017, input 640×640 [5].*

---

## 4.2. YOLOv6 (2022) — Industrial Deployment

**Paper:** "YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications" [6]  
**Tác giả:** Meituan Vision AI

### 4.2.1. Re-parameterization — Kỹ thuật chủ đạo

Kiến trúc **khác nhau giữa training và inference**:

- **Training:** Multi-branch — Conv $3 \times 3$ + Conv $1 \times 1$ + Identity song song → học phong phú
- **Inference:** Gộp (re-parameterize) thành **1 Conv $3 \times 3$ duy nhất** → nhanh
- **Toán:** $W_{\text{merged}} = W_{3 \times 3} + \text{pad}(W_{1 \times 1}) + I$

> **Ẩn dụ:** Giống **ôn thi** — khi ôn, bạn đọc 5 cuốn sách, ghi chú, làm bài tập (multi-branch). Khi vào phòng thi, bạn chỉ mang **1 tờ tóm tắt** chứa tinh hoa từ 5 cuốn (single branch).

[📌 YÊU CẦU CHÈN ẢNH: Tác giả hãy chèn sơ đồ Re-parameterization (training multi-branch → inference single-branch) từ paper YOLOv6 [6].]

### 4.2.2. Kiến trúc chi tiết

| Thành phần | Kỹ thuật | Giải thích |
|:--|:--|:--|
| Backbone | EfficientRep | RepVGG blocks, re-parameterizable |
| Neck | Rep-PAN | PAN + RepVGG blocks |
| Head | Efficient Decoupled | Tách cls/reg (giống YOLOX), giảm 1 Conv |
| Assignment | SimOTA → TAL | Task-Aligned Learning |
| Loss | VFL + SIoU/GIoU + DFL | Varifocal Loss cho classification |
| Anchor | **Anchor-free** | FCOS-style, 4 distances từ tâm |

### 4.2.3. Quantization Pipeline — Điểm mạnh nhất

INT8 chỉ mất **0.2% mAP** nhưng tăng **75% FPS** [6]. Dùng PTQ + QAT + RepOptimizer.

### 4.2.4. Kết quả

| Model | mAP@50:95 | FPS (T4 TRT FP16) | Params | So với v5 |
|:--|:-:|:-:|:-:|:--|
| YOLOv5-N | 28.0% | 602 | 1.9M | — |
| **YOLOv6-N** | **37.5%** | **1187** | **4.7M** | +9.5%, nhanh 2× |
| YOLOv5-S | 37.4% | 349 | 7.2M | — |
| **YOLOv6-S** | **45.0%** | **495** | **18.5M** | +7.6% |
| YOLOv5-L | 49.0% | 113 | 46.5M | — |
| **YOLOv6-L** | **52.8%** | **116** | **59.6M** | +3.8% |

*Ghi chú: FPS đo trên T4 GPU, TensorRT FP16. Không so sánh trực tiếp với FPS đo trên V100 hoặc Titan X ở thế hệ trước [6].*

---

## 4.3. YOLOv7 (2022) — SOTA Accuracy

**Paper:** "YOLOv7: Trainable Bag-of-Freebies Sets New SOTA for Real-Time Object Detectors" (CVPR 2023) [7]  
**Tác giả:** Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao

### 4.3.1. E-ELAN (Extended Efficient Layer Aggregation Network)

Mở rộng ELAN bằng cách tăng **cardinality** (số nhánh song song) mà **không phá gradient path gốc** [7].

> **Ẩn dụ:** ELAN giống 1 dây chuyền sản xuất với 3 công nhân nối tiếp. E-ELAN thêm "ca kíp" mới (expand cardinality) nhưng giữ nguyên dây chuyền cũ, rồi **shuffle + merge** kết quả → output phong phú hơn.

### 4.3.2. Planned Re-parameterized Conv

RepConv (RepVGG) **không nên dùng** khi có identity connection → xung đột gradient. **RepConvN:** Bỏ identity branch, chỉ giữ conv $1 \times 1$ + conv $3 \times 3$. Tăng 0.5% AP [7].

### 4.3.3. Coarse-to-Fine Label Assignment

- **Lead Head** (fine-grained): Gán nhãn chặt chẽ — chính xác cao
- **Auxiliary Head** (coarse-grained): Gán nhãn lỏng hơn → nhiều positive samples → train tốt hơn. **Chỉ dùng khi training, bỏ khi inference** → zero cost [7]

> **Ẩn dụ:** Lead head giống **giáo viên chấm thi nghiêm khắc** (chỉ cho điểm khi đúng hoàn toàn). Auxiliary head giống trợ giảng nhẹ nhàng (cho điểm khi gần đúng) → sinh viên nhận nhiều feedback → học tốt hơn.

### 4.3.4. Kết quả

**Base models (V100 GPU):**

| Model | Params | FPS | mAP@50:95 | So sánh |
|:--|:-:|:-:|:-:|:--|
| YOLOv5-L | 46.5M | 99 | 48.8% | — |
| YOLOX-L | 54.2M | 68 | 50.0% | — |
| PPYOLOE-L | 52.2M | 78 | 51.4% | — |
| **YOLOv7** | **36.9M** | **161** | **51.4%** | Ít params nhất, nhanh nhất |
| **YOLOv7-X** | **71.3M** | **114** | **53.1%** | Vượt PPYOLOE |

**Large models — SOTA:**

| Model | FPS | mAP@50:95 |
|:--|:-:|:-:|
| YOLOv7-W6 | 84 | 54.9% |
| YOLOv7-E6 | 56 | 55.9% |
| **YOLOv7-E6E** | **36** | **56.8%** |
| SWIN-L Cascade R-CNN | 12 | 53.9% |

*Ghi chú: FPS đo trên V100 GPU. v7-E6E (56.8%) vượt tất cả detectors tại thời điểm đó, kể cả Transformer-based [7].*

---

## 4.4. YOLOv8 (2023) — Đa năng nhất

**Tác giả:** Ultralytics | *Không có paper chính thức* [8]

### 4.4.1. Ba thay đổi kiến trúc lớn so với v5

**1. Anchor-Free:** Bỏ hoàn toàn anchor boxes. Dự đoán **center point + 4 distances** (khoảng cách từ tâm đến 4 cạnh).

> **Ẩn dụ:** Anchor-based giống chọn **khung ảnh S/M/L** rồi chỉnh. Anchor-free giống **vẽ tự do** — đứng tại tâm vật, đo 4 hướng lên/xuống/trái/phải.

**2. Decoupled Head:** Tách classification và regression thành 2 branches riêng biệt [8].

**3. DFL (Distribution Focal Loss):** Thay vì dự đoán 1 giá trị cho mỗi tọa độ box, dự đoán **phân phối xác suất** trên 16 bins.

> **Ẩn dụ:** Thay vì nói "cạnh trái cách tâm 25px" (1 số duy nhất), DFL nói "xác suất 10% ở 24px, 70% ở 25px, 20% ở 26px" → phản ánh **mức độ không chắc chắn**, localization chính xác hơn.

### 4.4.2. C2f Module

Kế thừa C3 (v5) + cảm hứng ELAN (v7): **concat output của TẤT CẢ bottlenecks** thay vì chỉ output cuối → gradient flow phong phú hơn.

[📌 YÊU CẦU CHÈN ẢNH: Tác giả hãy chèn sơ đồ so sánh C3 (v5) vs C2f (v8), thể hiện gradient flow và feature concatenation.]

### 4.4.3. Multi-task — 6 tác vụ

| Tác vụ | Output | Ví dụ ứng dụng |
|:--|:--|:--|
| Detection | Bounding boxes | Phát hiện xe, người |
| Segmentation | Pixel-level masks | Tách nền ảnh |
| Classification | Class label | Phân loại sản phẩm |
| Pose Estimation | 17 Keypoints | Phân tích tư thế |
| OBB | Rotated boxes | Ảnh vệ tinh |
| Tracking | Box + Track ID | Đếm người qua cửa |

### 4.4.4. Kết quả

| Model | Params | mAP@50:95 | So với v5 |
|:--|:-:|:-:|:--|
| v8n | 3.2M | 37.3% | v5n: 28.0% (**+9.3%**) |
| v8s | 11.2M | 44.9% | v5s: 37.4% (**+7.5%**) |
| v8m | 25.9M | 50.2% | v5m: 45.4% (**+4.8%**) |
| v8l | 43.7M | 52.9% | v5l: 49.0% (**+3.9%**) |
| v8x | 68.2M | 53.9% | v5x: 50.7% (**+3.2%**) |

⚠️ **Lưu ý quan trọng:** v8-X (53.9%) **thua** v7-E6E (56.8%) về mAP thuần túy. Tuy nhiên v8 thắng ở: ecosystem (★★★★★), multi-task (6 tác vụ), API thống nhất [7][8].

**→ Nếu cần accuracy tối đa → v7. Nếu cần đa năng + dễ dùng → v8.**
