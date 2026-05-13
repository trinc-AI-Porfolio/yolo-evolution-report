# DỊCH CHI TIẾT PAPER YOLOv2 — "YOLO9000: Better, Faster, Stronger"
**Tác giả:** Joseph Redmon, Ali Farhadi
**Hội nghị:** CVPR 2017 | **Link:** https://arxiv.org/abs/1612.08242

---

## ABSTRACT (Tóm tắt)

Chúng tôi giới thiệu YOLO9000 — hệ thống phát hiện vật thể real-time tiên tiến nhất, có thể phát hiện hơn 9000 loại vật thể. Đầu tiên, chúng tôi đề xuất nhiều cải tiến cho phương pháp detection YOLO, bao gồm cả kỹ thuật mới và kỹ thuật từ các nghiên cứu trước. Model cải tiến, YOLOv2, đạt SOTA trên các tác vụ detection tiêu chuẩn như PASCAL VOC và COCO. Ở 67 FPS, YOLOv2 đạt 76.8 mAP trên VOC 2007. Ở 40 FPS, YOLOv2 đạt 78.6 mAP — vượt trội Faster R-CNN với ResNet và SSD trong khi vẫn chạy nhanh hơn đáng kể. Cuối cùng, chúng tôi đề xuất phương pháp joint training trên cả object detection và classification. Sử dụng phương pháp này, YOLO9000 được train đồng thời trên COCO detection dataset và ImageNet classification dataset. YOLO9000 dự đoán detections cho hơn 9000 loại vật thể khác nhau và vẫn chạy real-time.

---

## 1. INTRODUCTION (Giới thiệu)

Object detection phổ quát phải nhanh, chính xác, và có khả năng nhận biết nhiều loại vật thể. Kể từ khi neural networks ra đời, các detection frameworks ngày càng nhanh và chính xác. Tuy nhiên, hầu hết detection methods vẫn giới hạn ở một tập nhỏ vật thể.

Các detection datasets hiện tại nhỏ hơn nhiều so với classification datasets. Dataset detection phổ biến nhất chứa hàng nghìn đến hàng trăm nghìn ảnh với vài chục đến vài trăm tags. Classification datasets có hàng triệu ảnh với hàng chục hoặc hàng trăm nghìn categories.

Chúng tôi muốn detection mở rộng tới mức của object classification. Tuy nhiên, labelling ảnh cho detection tốn kém hơn nhiều so với classification (cần vẽ bounding boxes). Do đó, chúng tôi đề xuất phương pháp tận dụng lượng lớn classification data có sẵn và sử dụng nó để mở rộng phạm vi các detection systems hiện tại.

Phương pháp sử dụng phân cấp hierarchical view của object classification cho phép kết hợp các datasets classification khác nhau lại với nhau.

Chúng tôi cũng đề xuất joint training algorithm cho phép train object detectors trên cả detection và classification data. Phương pháp tận dụng labelled detection images để học localize vật thể chính xác, đồng thời sử dụng classification images để tăng vocabulary và robustness.

---

## 2. BETTER (Tốt hơn)

YOLOv1 mắc nhiều lỗi localization so với region proposal-based methods. Ngoài ra, YOLO có recall tương đối thấp so với các phương pháp khác. Do đó, chúng tôi tập trung cải thiện recall và localization trong khi duy trì classification accuracy.

Thay vì mở rộng mạng, chúng tôi đơn giản hóa network và giúp representation dễ học hơn. Các cải tiến:

### 2.1. Batch Normalization (Chuẩn hóa theo batch)

Thêm batch normalization trên tất cả lớp convolutional. Điều này giúp cải thiện đáng kể convergence, đồng thời đóng vai trò regularizer → loại bỏ nhu cầu dropout mà không bị overfitting. Kết quả: **+2% mAP**.

### 2.2. High Resolution Classifier (Phân loại độ phân giải cao)

YOLOv1 train classifier trên 224×224, sau đó tăng lên 448×448 cho detection. Model phải vừa chuyển sang tác vụ mới vừa điều chỉnh cho resolution mới.

YOLOv2: Finetune classification network ở **448×448** trước trên ImageNet (10 epochs). Cho network thời gian điều chỉnh filters cho high-resolution input. Sau đó finetune cho detection. Kết quả: **+4% mAP**.

### 2.3. Convolutional With Anchor Boxes

YOLOv1 dự đoán coordinates trực tiếp bằng fully connected layers. Faster R-CNN dự đoán offsets và confidences cho anchor boxes bằng hand-picked priors (RPN).

Chúng tôi bỏ fully connected layers, dùng anchor boxes để dự đoán bounding boxes. Thay đổi:
- Bỏ 1 pooling layer → tăng resolution output
- Thu nhỏ input từ 448 xuống **416** → feature map lẻ (13×13) → có 1 center cell (vật lớn thường nằm giữa ảnh)
- Dự đoán class và objectness cho mỗi anchor box (thay vì mỗi grid cell)

Kết quả: mAP giảm nhẹ (69.5% → 69.2%) nhưng **recall tăng 81% → 88%** — nhiều room hơn để cải thiện.

### 2.4. Dimension Clusters (Phân cụm kích thước)

Vấn đề: Anchor boxes trong Faster R-CNN được chọn tay → không tối ưu. Nếu chọn anchor tốt hơn → model dễ học hơn.

Giải pháp: Chạy **k-means clustering** trên bounding boxes của training data để tự động tìm anchor tốt.

Dùng distance metric đặc biệt (không dùng Euclidean vì box lớn tạo error lớn hơn):
> d(box, centroid) = 1 - IoU(box, centroid)

Chọn **k = 5** anchors → đạt IoU trung bình tương đương 9 anchors chọn tay. 5 anchors cluster tốt hơn vì tìm được prior phù hợp data hơn.

### 2.5. Direct Location Prediction (Dự đoán vị trí trực tiếp)

Vấn đề khi dùng anchor boxes với YOLO: model bất ổn, đặc biệt ở early iterations. Nguyên nhân: formulation của Faster R-CNN không ràng buộc → predicted box có thể nằm bất kỳ đâu trong ảnh, bất kể vị trí anchor.

Giải pháp: Dự đoán location **tương đối so với grid cell**, dùng **logistic activation (sigmoid)** để ràng buộc output trong khoảng [0, 1]:
- **bx = σ(tx) + cx** (cx = offset của cell từ góc trên trái)
- **by = σ(ty) + cy**
- **bw = pw × e^(tw)** (pw = width của anchor prior)
- **bh = ph × e^(th)**
- **Pr(object) × IoU = σ(to)**

Ràng buộc này giúp network ổn định hơn, dễ học hơn. Kết quả: **+5% mAP** so với dùng anchor boxes không ràng buộc.

### 2.6. Fine-Grained Features (Đặc trưng chi tiết)

YOLOv2 dự đoán trên feature map 13×13, đủ cho vật lớn nhưng khó cho vật nhỏ.

Giải pháp: Thêm **passthrough layer** — lấy feature map 26×26 từ lớp trước, reshape thành 13×13 bằng cách xếp các vùng 2×2 liền kề vào channels (26×26×512 → 13×13×2048), rồi concatenate với feature map 13×13 gốc.

Kết quả: **+1% mAP**.

### 2.7. Multi-Scale Training (Huấn luyện đa tỷ lệ)

YOLOv2 chỉ dùng conv + pooling → có thể chạy ở bất kỳ input size nào.

Mỗi **10 batches**, network random chọn input size mới: bội số của 32, từ {320, 352, ..., 608}.

Kết quả: cùng 1 model chạy ở nhiều resolutions:
- **288×288**: 69.0% mAP, >90 FPS (nhanh hơn Fast YOLO mà chính xác hơn)
- **416×416**: 76.8% mAP, 67 FPS
- **544×544**: 78.6% mAP, 40 FPS (vượt Faster R-CNN, SSD)

---

## 3. FASTER (Nhanh hơn)

### 3.1. Darknet-19

Kiến trúc backbone mới:
- **19 lớp convolutional + 5 maxpooling**
- Dùng 3×3 filters, double channels sau mỗi pooling
- Dùng 1×1 filters để nén representation giữa các 3×3
- Dùng batch normalization, global average pooling
- **5.58 billion FLOPs** (so với VGG-16: 30.69B FLOPs)
- ImageNet Top-1: 72.9%, Top-5: 91.2% (tương đương VGG nhưng nhanh hơn nhiều)

### 3.2. Training for Classification

Train trên ImageNet 1000 classes: 160 epochs, SGD (lr=0.1), polynomial rate decay (power=4), weight decay=0.0005, momentum=0.9. Augmentation: random crops, rotations, hue/saturation/exposure shifts.

Sau đó finetune ở 448×448 (10 epochs, lr=10⁻³). Top-1: 76.5%, Top-5: 93.3%.

### 3.3. Training for Detection

Thay đổi cho detection:
- Bỏ lớp conv cuối, thêm 3 lớp conv 3×3 (1024 filters mỗi lớp)
- Cuối cùng: lớp conv 1×1 với số outputs cần thiết
- Thêm passthrough layer từ lớp conv 3×3 cuối (512 filters) có feature map 26×26
- Train 160 epochs, lr bắt đầu 10⁻³, giảm 10× ở epoch 60 và 90
- Weight decay=0.0005, momentum=0.9, augmentation giống v1

---

## 4. STRONGER (Mạnh hơn) — YOLO9000

### 4.1. Hierarchical Classification (Phân loại phân cấp)

ImageNet labels dựa trên WordNet — đồ thị ngôn ngữ có cấu trúc. Chúng tôi đơn giản hóa bằng cách xây dựng **WordTree** — cây phân cấp từ các visual nouns trong ImageNet.

Cách xây dựng: Với mỗi label trong ImageNet, tìm đường đi lên gốc (root) qua WordNet. Nhiều synsets chỉ có 1 đường → thêm trực tiếp. Nếu nhiều đường → chọn đường ngắn nhất.

Kết quả: **WordTree** với 1000 labels từ ImageNet thành cây phân cấp. Mỗi node dự đoán conditional probability: Pr(category | parent category).

Dự đoán: nhân conditional probabilities từ gốc xuống:
> Pr(Norfolk terrier) = Pr(Norfolk terrier | terrier) × Pr(terrier | hunting dog) × Pr(hunting dog | dog) × ... × Pr(physical object)

Để classification, giả định Pr(physical object) = 1.

Kết quả: ImageNet Top-1 = 71.9% trên 1000 classes, so với 72.9% khi train flat. Nhỏ giảm vì model phải classify ở nhiều cấp granularity.

### 4.2. Dataset Combination with WordTree

Dùng WordTree để merge datasets. Map labels từ nhiều datasets vào cây. Ví dụ: ImageNet có "Norfolk terrier", COCO có "dog" → cả 2 đều nằm trên cây.

### 4.3. Joint Classification and Detection

Xây dựng WordTree kết hợp COCO và top 9000 classes từ ImageNet. Cây kết quả có **9418 categories**. ImageNet lớn hơn COCO nhiều → cân bằng bằng cách oversample COCO (tỷ lệ 4:1).

Training:
- Khi model thấy **detection image** (COCO): backprop loss đầy đủ (classification + localization)
- Khi model thấy **classification image** (ImageNet): chỉ backprop classification loss ở phần cây tương ứng

### 4.4. Kết quả YOLO9000

- ImageNet detection task (200 classes): **19.7% mAP** tổng thể
- Trên 44 classes có trong cả COCO: 16.0% mAP
- Trên 156 classes KHÔNG có detection data: model vẫn detect được nhờ hierarchical prediction

Phân tích: YOLO9000 detect tốt các loại động vật mới vì COCO có nhiều data động vật → generalize tốt. Yếu hơn ở categories như quần áo, thiết bị vì COCO thiếu data tương tự.

---

## 5. CONCLUSION (Kết luận)

YOLOv2 là detector real-time SOTA, nhanh hơn và chính xác hơn các detection systems khác trên nhiều detection benchmarks.

YOLOv2 có thể chạy ở nhiều resolutions → trade-off linh hoạt giữa tốc độ và accuracy. Ở 67 FPS, YOLOv2 đạt 76.8% mAP trên VOC 2007. Ở 40 FPS, đạt 78.6% mAP — vượt Faster R-CNN với ResNet và SSD.

YOLO9000 là framework real-time cho detection hơn 9000 object categories bằng cách tận dụng joint training trên detection và classification data sử dụng WordTree.

---

## BẢNG TỔNG KẾT CẢI TIẾN YOLOv2

| # | Kỹ thuật | mAP (%) | Thay đổi |
|---|---------|---------|---------|
| Baseline (v1) | - | 63.4 | - |
| 1 | + Batch Normalization | 65.4 | +2.0 |
| 2 | + High-Res Classifier | 69.4 | +4.0 |
| 3 | + Convolutional (bỏ FC) | 69.2 | -0.2 |
| 4 | + Anchor Boxes | - | Recall +7% |
| 5 | + Dimension Clusters | - | IoU tốt hơn |
| 6 | + Direct Location | 74.4 | +5.0 |
| 7 | + Passthrough | 75.4 | +1.0 |
| 8 | + Multi-Scale | 76.8 | +1.4 |
| **Tổng** | **YOLOv2 (416×416)** | **76.8** | **+13.4** |

---

*Bản dịch chi tiết từ paper gốc arXiv:1612.08242. Mọi nội dung kỹ thuật giữ nguyên ý nghĩa.*
