# KHẢO SÁT TOÀN DIỆN SỰ PHÁT TRIỂN CỦA YOLO
## Từ YOLOv1 (2015) đến YOLO26 (2026): Kiến trúc — Phương pháp — Kết quả — Xu hướng

**Tác giả:** Nguyễn Cao Trị  
**Ngày biên soạn:** 14/05/2026

---

# MỤC LỤC

1. [Mở đầu](#chương-1-mở-đầu)
2. [Nền tảng Lý thuyết](#chương-2-nền-tảng-lý-thuyết)
3. [Thế hệ Đầu: YOLOv1–v4 (2015–2020)](#chương-3-thế-hệ-đầu-yolov1v4)
4. [Thế hệ Giữa: YOLOv5–v8 (2020–2023)](#chương-4-thế-hệ-giữa-yolov5v8)
5. [Thế hệ Mới: YOLOv9–v12 (2024–2025)](#chương-5-thế-hệ-mới-yolov9v12)
6. [Mở rộng: YOLOE & YOLO26](#chương-6-mở-rộng-yoloe--yolo26)
7. [Các biến thể YOLO](#chương-7-các-biến-thể-yolo)
8. [So sánh Tổng hợp & Đánh giá](#chương-8-so-sánh-tổng-hợp--đánh-giá)
9. [Kết luận & Khuyến nghị](#chương-9-kết-luận--khuyến-nghị)
10. [Phụ lục: Bảng Thuật ngữ](#phụ-lục-bảng-thuật-ngữ)
11. [Tài liệu Tham khảo](#tài-liệu-tham-khảo)

---

# Chương 1: MỞ ĐẦU

## 1.1. Giới thiệu bài toán Object Detection

Phát hiện vật thể (Object Detection) là một trong những bài toán trung tâm của thị giác máy tính (Computer Vision). Mục tiêu là xác định **vị trí** (bounding box) và **loại** (class) của tất cả vật thể trong ảnh hoặc video. Lĩnh vực này được ứng dụng rộng rãi trong xe tự hành, giám sát an ninh, y tế, công nghiệp và nông nghiệp.

Trước khi YOLO ra đời (2015), các phương pháp detection chủ đạo là **two-stage detectors**:

- **R-CNN** (2014): Dùng Selective Search tạo ~2000 vùng đề xuất (region proposals), mỗi vùng qua CNN trích features, rồi SVM phân loại. Cực chậm: **>40 giây/ảnh**.
- **Fast R-CNN** (2015): Cải tiến bằng cách chia sẻ CNN features cho tất cả vùng, nhưng vẫn phụ thuộc Selective Search.
- **Faster R-CNN** (2015): Thay Selective Search bằng Region Proposal Network (RPN) học được, nhưng vẫn là pipeline **2 giai đoạn rời rạc**.

> **Ẩn dụ:** Các phương pháp two-stage giống như **bảo vệ tòa nhà kiểm tra 2 lần**: lần 1 liếc qua đánh dấu 2000 người "nghi ngờ", lần 2 gọi từng người kiểm tra căn cước → chính xác nhưng cực chậm.

## 1.2. YOLO — Bước ngoặt lịch sử

**YOLO (You Only Look Once)** ra đời năm 2015, thay đổi hoàn toàn cách tiếp cận: coi object detection là **bài toán hồi quy duy nhất** (single regression problem) — từ pixel ảnh trực tiếp ra bounding boxes và class probabilities, chỉ qua **một lần forward pass** → real-time (45 FPS) [1].

> **Ẩn dụ:** YOLO giống bạn **liếc mắt 1 cái** vào phòng: "Góc trái có cái bàn, giữa có con mèo, bên phải có cái ghế" — biết ngay **cái gì + ở đâu**, chỉ qua 1 lần nhìn.

Từ 2015 đến 2026, YOLO đã trải qua **14 phiên bản chính** (v1–v12, YOLOE, YOLO26) và nhiều biến thể, với mAP tăng từ 63.4% (VOC) lên ~57.2% (COCO — benchmark khó hơn nhiều), tốc độ từ 45 FPS lên hàng trăm FPS.

## 1.3. Mục tiêu và phạm vi báo cáo

**Mục tiêu:** Khảo sát toàn diện sự phát triển của YOLO qua các đời, phân tích kiến trúc, phương pháp, kết quả, ưu/nhược điểm, và xu hướng tương lai.

**Phạm vi:**
- Dòng chính: YOLOv1 → YOLOv12, YOLOE, YOLO26
- Biến thể: YOLOX, PP-YOLOE, YOLO-NAS, YOLO-World
- So sánh đa chiều và khuyến nghị lựa chọn

---

# Chương 2: NỀN TẢNG LÝ THUYẾT

## 2.1. Pipeline Object Detection hiện đại

Một detector hiện đại gồm 3 phần:

| Thành phần | Chức năng | Ví dụ |
|:--|:--|:--|
| **Backbone** | Trích xuất features từ ảnh | Darknet, CSPDarknet, ResNet |
| **Neck** | Kết hợp features ở nhiều scales | FPN, PAN, BiFPN |
| **Head** | Dự đoán bounding boxes và classes | Coupled / Decoupled head |

> **Ẩn dụ:** Backbone giống **mắt** (nhìn và nhận diện đặc trưng), Neck giống **não bộ** (tổng hợp thông tin từ nhiều góc nhìn), Head giống **miệng** (nói ra đáp án cuối cùng).

[📌 YÊU CẦU CHÈN ẢNH: Tác giả hãy chèn sơ đồ khối tổng quát Backbone → Neck → Head của một YOLO detector hiện đại.]

## 2.2. One-stage vs Two-stage Detectors

| Tiêu chí | Two-stage (Faster R-CNN) | One-stage (YOLO, SSD) |
|:--|:--|:--|
| Tốc độ | Chậm (5–20 FPS) | Nhanh (30–300 FPS) |
| Accuracy | Cao hơn (trước 2020) | Bắt kịp từ YOLOv4+ [4] |
| Pipeline | 2 giai đoạn rời rạc | End-to-end |

## 2.3. Các khái niệm cốt lõi

### 2.3.1. IoU (Intersection over Union)

Đo mức trùng lặp giữa predicted box (B_p) và ground truth (B_gt):

$$\text{IoU} = \frac{|B_p \cap B_{gt}|}{|B_p \cup B_{gt}|}$$

> **Ẩn dụ:** Tưởng tượng 2 tờ giấy đặt chồng lên nhau. IoU = *diện tích phần chồng* chia cho *tổng diện tích cả 2 tờ (đếm 1 lần)*. IoU = 1.0 nghĩa là 2 tờ trùng khít hoàn hảo.

### 2.3.2. mAP (mean Average Precision)

- **mAP@50:** Tính Average Precision ở ngưỡng IoU ≥ 0.5. Metric chính trên PASCAL VOC.
- **mAP@50:95:** Trung bình AP ở IoU từ 0.5 đến 0.95 (bước 0.05) — metric chính trên MS COCO, khắt khe hơn nhiều.

### 2.3.3. Anchor Boxes

Các hộp mẫu (templates) được định nghĩa trước với kích thước và tỷ lệ khác nhau. Model dự đoán **offsets** so với anchors thay vì tọa độ tuyệt đối.

> **Ẩn dụ:** Anchor giống đi **may đồ chọn size S/M/L có sẵn**, rồi chỉ cần sửa lai cho vừa — nhanh và chính xác hơn so với đo từ đầu (anchor-free).

### 2.3.4. NMS (Non-Maximum Suppression)

Thuật toán loại bỏ bounding boxes trùng lặp: giữ box có confidence cao nhất, bỏ các boxes có IoU lớn với box đã chọn.

### 2.3.5. Tiến hóa Loss Function

| Loss | Phiên bản | Đặc điểm |
|:--|:--|:--|
| MSE/SSE | v1–v3 | Đơn giản, coi box lớn/nhỏ ngang nhau |
| GIoU | v6 | Xét cả vùng bao ngoài khi 2 box không giao |
| CIoU | v4–v8 | + khoảng cách tâm + tỷ lệ khung hình |
| DFL | v8–v12 | Dự đoán phân phối xác suất thay vì giá trị đơn |
| Bỏ DFL | YOLO26 | Quay về dự đoán trực tiếp, bù bằng training tricks |

### 2.3.6. Datasets chuẩn

- **PASCAL VOC** (2007/2012): 20 classes, ~10K ảnh. Metric: mAP@50.
- **MS COCO:** 80 classes, 330K ảnh. Metric: mAP@50:95. Benchmark chính từ v3+.
- **LVIS:** 1203 categories. Dùng cho open-vocabulary evaluation (YOLOE) [13].

---

# Chương 3: THẾ HỆ ĐẦU — YOLOv1–v4 (2015–2020)

## 3.1. YOLOv1 (2015) — Khởi nguồn

**Paper:** "You Only Look Once: Unified, Real-Time Object Detection" (CVPR 2016) [1]  
**Tác giả:** Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi

### 3.1.1. Ý tưởng cốt lõi

Chia ảnh thành lưới **S × S** (S=7), mỗi ô dự đoán **B=2** bounding boxes và **C=20** class probabilities. Toàn bộ pipeline là **một mạng neural duy nhất**, end-to-end.

### 3.1.2. Kiến trúc

24 lớp convolutional + 2 lớp fully connected. Output: tensor **7 × 7 × 30**.

[📌 YÊU CẦU CHÈN ẢNH: Tác giả hãy chèn sơ đồ kiến trúc YOLOv1 từ paper gốc (Figure 3, Redmon et al. 2016) [1].]

### 3.1.3. Hàm Loss (5 thành phần)

$$\mathcal{L} = \lambda_{\text{coord}} \sum_{i=0}^{S^2}\sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right]$$

$$+ \sum_{i}\sum_{j} \mathbb{1}_{ij}^{\text{obj}}(C_i - \hat{C}_i)^2 + \lambda_{\text{noobj}} \sum_{i}\sum_{j} \mathbb{1}_{ij}^{\text{noobj}}(C_i - \hat{C}_i)^2 + \sum_{i} \mathbb{1}_{i}^{\text{obj}} \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2$$

Trong đó: λ_coord = 5, λ_noobj = 0.5. Dùng √w, √h để **giảm ảnh hưởng sai số** ở box lớn [1].

### 3.1.4. Kết quả

| Model | mAP (VOC 2007) | FPS |
|:--|:-:|:-:|
| Fast YOLO | 52.7% | 155 |
| YOLO | 63.4% | 45 |
| YOLO VGG-16 | 66.4% | 21 |
| Faster R-CNN | 70.0% | 0.5 |

*Ghi chú: FPS đo trên Titan X (Maxwell). Không so sánh trực tiếp với FPS thế hệ GPU sau.*

**Ưu điểm:** Real-time đầu tiên; ít false positive background; tổng quát hóa tốt (Picasso dataset: 53.3% vs R-CNN 10.2%) [1].  
**Nhược điểm:** Vật nhỏ kém (grid thô 7×7); localization sai nhiều; mỗi ô chỉ dự đoán 1 class.

---

## 3.2. YOLOv2 / YOLO9000 (2016) — Better, Faster, Stronger

**Paper:** "YOLO9000: Better, Faster, Stronger" (CVPR 2017) [2]  
**Tác giả:** Joseph Redmon, Ali Farhadi

### 3.2.1. Cải tiến tích lũy

| # | Kỹ thuật | mAP (%) | Thay đổi |
|:-:|:--|:-:|:--|
| 0 | Baseline (v1) | 63.4 | — |
| 1 | + Batch Normalization | 65.4 | +2.0 |
| 2 | + High-Res Classifier (448) | 69.4 | +4.0 |
| 3 | + Anchor Boxes (5 anchors) | 69.2 | Recall +7% |
| 4 | + Dimension Clusters (k-means) | — | IoU tốt hơn |
| 5 | + Direct Location (sigmoid) | 74.4 | +5.0 |
| 6 | + Passthrough (26×26→13×13) | 75.4 | +1.0 |
| 7 | + Multi-Scale Training | 76.8 | +1.4 |
| | **YOLOv2 (416×416)** | **76.8** | **+13.4** |

### 3.2.2. Darknet-19 & YOLO9000

Backbone mới: 19 conv + 5 maxpool, chỉ **5.58B FLOPs** (VGG-16: 30.69B — nhanh gấp 5×). YOLO9000 dùng **WordTree** joint training trên COCO + ImageNet → detect **9000+ categories** [2].

---

## 3.3. YOLOv3 (2018) — Multi-Scale Detection

**Paper:** "YOLOv3: An Incremental Improvement" (Tech Report) [3]  
**Tác giả:** Joseph Redmon, Ali Farhadi *(paper cuối cùng của Redmon trước khi dừng nghiên cứu CV vì lo ngại đạo đức)*

### 3.3.1. Đột phá: 3 Scales

Dự đoán ở **3 scales** (13×13, 26×26, 52×52) theo FPN-style, mỗi scale 3 anchors = 9 anchors tổng. Thay Softmax bằng **Sigmoid** cho multi-label classification [3].

> **Ẩn dụ:** v1–v2 nhìn qua 1 ống kính. v3 nhìn qua **3 ống kính zoom khác nhau** cùng lúc: zoom xa (vật lớn), trung bình, zoom gần (vật nhỏ).

### 3.3.2. Darknet-53

53 conv layers + residual connections. Chính xác bằng ResNet-152 nhưng **nhanh gấp 2×** (78 vs 37 FPS) [3].

**Kết quả:** mAP@50: 57.9% (COCO), ngang RetinaNet nhưng nhanh gấp 3.8×. mAP@50:95: 33.0%.

---

## 3.4. YOLOv4 (2020) — Tập đại thành

**Paper:** "YOLOv4: Optimal Speed and Accuracy of Object Detection" [4]  
**Tác giả:** Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao

### 3.4.1. Phương pháp luận: BoF & BoS

| Nhóm | Ý nghĩa | Kỹ thuật tiêu biểu |
|:--|:--|:--|
| **Bag of Freebies** | Chỉ tăng training cost | Mosaic, CutMix, CIoU Loss, Label Smoothing, SAT |
| **Bag of Specials** | Tăng nhẹ inference cost | Mish activation, SPP, SAM, PAN, DIoU-NMS |

> **Ẩn dụ:** BoF giống **đầu bếp gom tất cả công thức hay nhất thế giới** vào 1 cuốn sách. Không cần phát minh mới, chỉ cần biết chọn và kết hợp đúng.

### 3.4.2. Kiến trúc: CSPDarknet53 + SPP + PAN

- **CSP (Cross Stage Partial):** Chia features thành 2 phần, chỉ 1 phần qua processing → giảm **20% computation** mà giữ accuracy.
- **SPP:** MaxPool nhiều kích thước → tăng receptive field.
- **PAN:** Bottom-up pathway bổ sung cho FPN → localization signals mạnh hơn.

[📌 YÊU CẦU CHÈN ẢNH: Tác giả hãy chèn sơ đồ kiến trúc YOLOv4 (CSPDarknet53 + SPP + PAN) từ paper gốc [4].]

### 3.4.3. Kỹ thuật nổi bật

**Mosaic Augmentation:** Ghép 4 ảnh ngẫu nhiên thành 1 ảnh → model thấy nhiều context, giảm batch size cần thiết [4].

**CIoU Loss:**

$$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2} + \alpha v$$

Trong đó: ρ = khoảng cách Euclid giữa 2 tâm, c = đường chéo vùng bao, v = hệ số aspect ratio, α = hệ số cân bằng [4].

> **Ẩn dụ:** IoU chỉ đo "2 box chồng bao nhiêu %". CIoU đo thêm "2 tâm box cách nhau bao xa" và "hình dạng 2 box có giống nhau không" → toàn diện hơn.

### 3.4.4. Kết quả

**43.5% mAP** (COCO), ~65 FPS (V100). Nhảy vọt **+10.5%** so với v3 — bước nhảy lớn nhất lịch sử YOLO [4]. Train được trên **single GPU** (8–16 GB).

*Ghi chú: FPS đo trên V100 GPU. Không so sánh trực tiếp với FPS đo trên Titan X (v1–v3).*
