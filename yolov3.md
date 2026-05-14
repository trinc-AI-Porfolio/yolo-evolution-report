# DỊCH CHI TIẾT PAPER YOLOv3 — "YOLOv3: An Incremental Improvement"
**Tác giả:** Joseph Redmon, Ali Farhadi
**Năm:** 2018 | **Link:** https://arxiv.org/abs/1804.02767

*Lưu ý: Paper này rất ngắn (6 trang), viết theo phong cách casual/hài hước — đặc trưng của Redmon. Đây là paper cuối cùng của ông trong lĩnh vực CV.*

---

## ABSTRACT

Chúng tôi trình bày một số cập nhật cho YOLO. Thực hiện một vài thay đổi thiết kế để cải thiện model. YOLOv3 có kích thước lớn hơn nhưng chính xác hơn, vẫn nhanh. Ở 320×320 chạy trong 22ms với 28.2 mAP — chính xác bằng SSD nhưng nhanh hơn 3 lần. Khi sử dụng metric mAP50 cũ, YOLOv3 khá ấn tượng: đạt 57.9% mAP50 trong 51ms trên Titan X, so với RetinaNet 57.5% mAP50 trong 198ms — nhanh gấp 3.8 lần.

---

## 1. GIỚI THIỆU

*(Redmon viết hài hước:)* "Đôi khi bạn chỉ kiểu... kệ nó, mình ngồi xem Twitter 1 năm thay vì làm nghiên cứu."

Không có nhiều thay đổi lớn, chỉ là tập hợp các cải tiến nhỏ làm model tốt hơn. Cũng đã thử những thứ không hiệu quả.

---

## 2. CÁC THAY ĐỔI (The Deal)

### 2.1. Bounding Box Prediction

YOLOv3 dự đoán bounding boxes dùng dimension clusters làm anchor boxes (giống v2). Network dự đoán 4 coordinates cho mỗi bounding box: tx, ty, tw, th.

Nếu cell offset từ góc trên trái ảnh là (cx, cy) và anchor box có width pw, height ph:
- **bx = σ(tx) + cx**
- **by = σ(ty) + cy**
- **bw = pw × e^(tw)**
- **bh = ph × e^(th)**

Training dùng **sum of squared error loss**. Nếu ground truth cho coordinate là t̂ thì gradient = ground truth value − prediction: t̂ − t.

YOLOv3 dự đoán objectness score cho mỗi bounding box bằng **logistic regression**. Score = 1 cho anchor box có IoU cao nhất với ground truth object (chỉ gán 1 anchor/object). Nếu anchor box không phải best nhưng IoU > threshold (0.5) → bỏ qua prediction (không tính loss). Mỗi object chỉ được gán 1 bounding box predictor. Nếu anchor không được gán → không có loss cho coordinate hoặc class predictions, chỉ có objectness loss.

### 2.2. Class Prediction — Thay đổi quan trọng

**Bỏ softmax** → dùng **independent logistic classifiers** và **binary cross-entropy loss** cho mỗi class.

Lý do: Khi chuyển sang datasets phức tạp hơn như Open Images Dataset, có nhiều labels chồng chéo (ví dụ: "Woman" và "Person"). Softmax giả định các classes loại trừ lẫn nhau — điều này không phù hợp khi 1 vật có thể thuộc nhiều labels.

→ **Multi-label classification**: Cho phép 1 bounding box có nhiều class cùng lúc.

### 2.3. Predictions Across Scales — Đột phá chính

YOLOv3 dự đoán boxes ở **3 scales khác nhau**, trích xuất features ở 3 scales tương tự Feature Pyramid Networks (FPN).

**Cách hoạt động:**
1. Từ base feature extractor, thêm nhiều conv layers → dự đoán tensor 3D (bounding boxes, objectness, class predictions)
2. Lấy feature map từ 2 layers trước, **upsample 2×**
3. Concatenate với feature map có resolution tương ứng từ sớm hơn trong network (skip connection)
4. Thêm conv layers → dự đoán tensor tương tự
5. Lặp lại 1 lần nữa cho scale thứ 3

Kết quả: Dự đoán ở 3 scales:
- **Scale 1 (13×13):** 3 anchors lớn → vật lớn
- **Scale 2 (26×26):** 3 anchors trung → vật trung bình
- **Scale 3 (52×52):** 3 anchors nhỏ → vật nhỏ

Tổng: **9 anchors** (3 anchors × 3 scales), được tìm bằng k-means clustering.

Mỗi scale dự đoán tensor: **N × N × [3 × (4 + 1 + 80)]**
- 3 boxes/cell
- 4 bounding box offsets
- 1 objectness prediction
- 80 class predictions (COCO)

### 2.4. Feature Extractor — Darknet-53

Mạng mới kết hợp Darknet-19 + residual connections:
- **53 lớp convolutional**
- Dùng successive 3×3 và 1×1 conv layers
- Có **shortcut connections** (residual blocks)

So sánh:

| Backbone | Top-1 | Top-5 | Bn Ops | FPS |
|----------|-------|-------|--------|-----|
| Darknet-19 | 74.1 | 91.8 | 7.29 | 171 |
| ResNet-101 | 77.1 | 93.7 | 19.7 | 53 |
| ResNet-152 | 77.6 | 93.8 | 29.4 | 37 |
| **Darknet-53** | **77.2** | **93.8** | **18.7** | **78** |

Darknet-53: Chính xác bằng ResNet-152 nhưng **nhanh hơn 2×**. Hiệu quả nhất về FLOPs/accuracy.

### 2.5. Training

- Vẫn train full images, không hard negative mining
- Multi-scale training, data augmentation
- Batch normalization, Leaky ReLU
- Không dùng focal loss (thử rồi nhưng không hiệu quả)

---

## 3. KẾT QUẢ (How We Do)

### Bảng kết quả COCO:

| Phương pháp | mAP50 | mAP50-95 | Thời gian |
|------------|-------|----------|----------|
| SSD321 | 45.4% | 28.0% | 61ms |
| SSD513 | 50.4% | 31.2% | 125ms |
| DSSD321 | 46.1% | 28.0% | 85ms |
| RetinaNet-50-500 | 50.9% | 32.5% | 73ms |
| RetinaNet-101-800 | 57.5% | 37.8% | 198ms |
| **YOLOv3-320** | **51.5%** | **28.2%** | **22ms** |
| **YOLOv3-416** | **55.3%** | **31.0%** | **29ms** |
| **YOLOv3-608** | **57.9%** | **33.0%** | **51ms** |

**Nhận xét:**
- YOLOv3 608: 57.9% mAP50 — **ngang RetinaNet nhưng nhanh gấp 3.8×**
- Tuy nhiên, ở metric mAP50-95 (đánh giá IoU chặt hơn), YOLOv3 thua: 33.0% vs 37.8% → YOLO vẫn localize kém hơn ở IoU cao
- Cải tiến so với v2: mAP50 từ ~44% lên 57.9% → **bước nhảy lớn**

---

## 4. NHỮNG THỨ ĐÃ THỬ NHƯNG THẤT BẠI (Things We Tried That Didn't Work)

### 4.1. Anchor box x, y offset predictions
Thử dự đoán x, y offset theo bội số của width/height (giống Faster R-CNN). → Model **bất ổn, không hoạt động tốt**.

### 4.2. Linear x, y prediction thay sigmoid
Thử dùng linear activation cho x, y thay sigmoid. → mAP **giảm vài điểm**.

### 4.3. Focal Loss
Thử áp dụng focal loss (từ RetinaNet). → mAP **giảm khoảng 2 điểm**. Có thể vì YOLOv3 đã xử lý vấn đề focal loss giải quyết (class imbalance) bằng cách khác: tách objectness prediction + conditional class prediction.

### 4.4. Dual IoU thresholds and truth assignment
Thử dùng 2 ngưỡng IoU giống Faster R-CNN: IoU > 0.7 = positive, IoU < 0.3 = negative. → Kết quả **không tốt hơn**.

---

## 5. Ý NGHĨA CỦA CÁC METRICS (What This All Means)

YOLOv3 hoạt động tốt. Ở metric COCO mAP "cũ" (mAP50), nó cực mạnh và nhanh hơn tất cả. Nhưng ở metric mới (mAP50-95), hiệu suất giảm → cho thấy YOLOv3 gặp khó khăn khi cần bounding boxes **rất chính xác** (IoU > 0.75).

Redmon bình luận: "Metric COCO mới nhấn mạnh precision cao ở IoU. Nhưng tại sao? Nhiều ứng dụng không cần IoU cao như vậy." Ông đặt câu hỏi liệu metric này có phản ánh đúng nhu cầu thực tế không.

---

## 6. REBUTTAL (Phản biện — phần không thường thấy trong paper)

*(Redmon viết phần này rất hài hước, đáp trả reviewers)*

Khi được hỏi "What's new?", ông trả lời: "Bạn mong đợi gì từ 1 tech report? Đây không phải paper CVPR." Ông cũng nhấn mạnh rằng YOLOv3 có Darknet-53 hoàn toàn mới và multi-scale predictions mới.

---

## 7. KẾT LUẬN — Lời chia tay

*(Phần này rất đặc biệt — Redmon bày tỏ lo ngại đạo đức)*

Redmon viết: Nghiên cứu CV có nhiều ứng dụng tốt nhưng cũng có mặt tối. Giám sát hàng loạt, quân sự hóa AI đáng lo ngại. Ông tự hỏi liệu có nên tiếp tục public nghiên cứu không khi biết nó sẽ bị lạm dụng.

> "Với tư cách nhà nghiên cứu, chúng ta có trách nhiệm ít nhất là xem xét tác hại mà công việc của mình có thể gây ra và nghĩ cách giảm thiểu — đặc biệt khi công việc liên quan trực tiếp đến ứng dụng thực tế."

**Đây là paper cuối cùng của Joseph Redmon. Sau paper này, ông tuyên bố dừng nghiên cứu computer vision.**

---

## BẢNG TÓM TẮT CẢI TIẾN v3 so với v2

| Kỹ thuật | YOLOv2 | YOLOv3 |
|---------|--------|--------|
| Backbone | Darknet-19 (19 layers) | **Darknet-53** (53 layers + residual) |
| Scales | 1 scale (13×13) | **3 scales** (13, 26, 52) |
| Anchors | 5 anchors | **9 anchors** (3/scale) |
| Classification | Softmax | **Sigmoid** (multi-label) |
| Feature fusion | Passthrough 1 layer | **FPN-style** upsample + concat |
| mAP50 (COCO) | ~44% | **57.9%** |
| mAP50-95 | ~21% | **33.0%** |

---

*Bản dịch chi tiết từ paper gốc arXiv:1804.02767.*
