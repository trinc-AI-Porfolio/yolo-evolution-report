# DỊCH CHI TIẾT PAPER YOLOv4
**"YOLOv4: Optimal Speed and Accuracy of Object Detection"**
**Tác giả:** Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao
**Năm:** 2020 | **Link:** https://arxiv.org/abs/2004.10934

---

## ABSTRACT

Có rất nhiều features được cho là cải thiện accuracy của CNN: Batch Normalization, residual connections, data augmentation, v.v. Paper này kiểm nghiệm xem sự kết hợp nào của các features đó hoạt động tốt nhất, và giải thích lý thuyết tại sao. Một số features chỉ phù hợp cho một số models/problems nhất định, hoặc chỉ cho datasets nhỏ, trong khi một số khác phổ quát. Chúng tôi đề xuất nhiều kỹ thuật mới: WRC (Weighted-Residual-Connections), CSP (Cross-Stage-Partial), CmBN (Cross mini-Batch Normalization), SAT (Self-Adversarial-Training), Mish activation, Mosaic data augmentation, DropBlock regularization, CIoU loss. Chúng tôi đạt SOTA: **43.5% mAP (65.7% mAP50) trên COCO** ở tốc độ real-time ~65 FPS trên Tesla V100.

---

## 1. INTRODUCTION

Hầu hết CNN-based detectors chỉ áp dụng được trên hệ thống recommendation, không phải real-time. Mục tiêu chính: tạo detector hoạt động nhanh trong production systems, tối ưu cho **parallel computations** (GPU), không phải cho low computational power (edge).

Đóng góp chính:
1. Phát triển detector hiệu quả, mạnh, phù hợp cho single GPU training
2. Xác minh ảnh hưởng của BoF và BoS methods
3. Sửa đổi các methods SOTA cho phù hợp single GPU training (CBN → CmBN, PAN → modified PAN, SAM)

---

## 2. RELATED WORK (Survey chi tiết)

### 2.1. Object Detection Models

Modern detector gồm 2 phần:
- **Backbone:** Pre-train trên ImageNet. VGG16, ResNet-50, ResNeXt, Darknet, CSPDarknet, EfficientNet...
- **Head:** Dự đoán classes và bounding boxes

Head chia 2 loại:
- **Two-stage (dense prediction):** R-CNN, Fast R-CNN, Faster R-CNN, R-FCN, Libra R-CNN
- **One-stage (dense prediction):** SSD, YOLO, RetinaNet, FCOS, CenterNet

Thêm **Neck** — layers giữa backbone và head, thu thập feature maps ở các stages:
- **FPN** (top-down pathway)
- **PAN** (bottom-up + top-down)
- **BiFPN, ASFF, SFAM**

### 2.2. Bag of Freebies (BoF)

Kỹ thuật **chỉ tăng training cost, KHÔNG tăng inference cost**:

**Data augmentation:**
- Photometric: brightness, contrast, hue, saturation, noise
- Geometric: random scaling, cropping, flipping, rotating
- **Bỏ 1 phần ảnh:** Random Erase, CutOut, Hide-and-Seek, GridMask
- **Trộn ảnh:** MixUp, CutMix, **Mosaic** (mới — ghép 4 ảnh)
- **Style Transfer GAN** cho augmentation

**Regularization:**
- DropOut, DropConnect, DropBlock (bỏ vùng liên tục thay vì random pixels)

**Loss function cho Bounding Box:**
- MSE (dùng trong v1) → IoU Loss → GIoU → DIoU → **CIoU** (xét overlap + khoảng cách tâm + aspect ratio)

**Khác:** Label Smoothing, Knowledge Distillation

### 2.3. Bag of Specials (BoS)

Kỹ thuật **tăng nhẹ inference cost nhưng tăng accuracy đáng kể**:

**Enhance receptive field:** SPP (Spatial Pyramid Pooling), ASPP, RFB

**Attention modules:** SE (Squeeze-and-Excitation), SAM (Spatial Attention Module), CBAM

**Feature integration:** FPN, PAN, BiFPN, ASFF, SFAM

**Activation functions:** ReLU, LReLU, PReLU, SELU, Swish, **Mish**

**Post-processing:** NMS, Soft-NMS, **DIoU-NMS**

---

## 3. METHODOLOGY (Phương pháp)

### 3.1. Lựa chọn kiến trúc

**Backbone:** CSPDarknet53
- CSP (Cross Stage Partial): Chia feature map thành 2 phần, chỉ 1 phần qua residual blocks
- Giảm computation ~20% mà giữ accuracy
- Tốt hơn ResNet vì gradient flow đa dạng hơn

**Neck:** SPP + PAN (modified)
- SPP: Pooling ở 3 sizes (5×5, 9×9, 13×13) → concatenate → tăng receptive field đáng kể
- PAN: Bottom-up path aggregation — cải thiện localization signals ở lower layers

**Head:** YOLOv3 head (anchor-based, 3 scales)

### 3.2. Lựa chọn BoF và BoS

**BoF cho backbone:**
- CutMix, Mosaic augmentation
- DropBlock regularization
- Class label smoothing

**BoS cho backbone:**
- **Mish activation:** f(x) = x · tanh(softplus(x)) = x · tanh(ln(1 + eˣ)). Mượt hơn ReLU/Leaky ReLU, cho phép thông tin nhỏ qua khi x < 0
- CSP (Cross Stage Partial connections)
- MiWRC (Multi-input Weighted Residual Connections)

**BoF cho detector:**
- **CIoU Loss** cho bounding box regression
- **CmBN** (Cross mini-Batch Normalization): Thu thập statistics từ nhiều mini-batches trước
- DropBlock regularization
- **Mosaic augmentation** (mới)
- **Self-Adversarial Training (SAT)** (mới)
- Eliminate grid sensitivity
- Nhiều anchors cho 1 ground truth
- **Cosine annealing scheduler**
- Optimal hyperparameters (genetic algorithms)
- Random training shapes

**BoS cho detector:**
- **Mish** activation
- **SPP-block**
- **SAM-block** (modified)
- **PAN** path-aggregation block (modified)
- **DIoU-NMS**

### 3.3. Kỹ thuật mới

**Mosaic Augmentation:**
Ghép 4 ảnh training ngẫu nhiên thành 1 ảnh. Ưu điểm:
- Model thấy nhiều context hơn (4 ảnh cùng lúc)
- Batch normalization tính statistics trên 4 ảnh khác nhau → ít cần large batch size
- Vật thể xuất hiện ở nhiều vị trí bất thường → model robust hơn

**Self-Adversarial Training (SAT):**
2 forward-backward stages:
1. Network sửa đổi ảnh đầu vào (tạo adversarial perturbation) để maximize loss — đánh lừa chính nó
2. Train network detect vật thể trên ảnh đã bị sửa đổi → model robust hơn

**CmBN:** Thu thập BN statistics từ toàn bộ mini-batch, không chỉ batch hiện tại.

**Modified SAM:** Thay spatial attention từ pointwise sang layerwise.

**Modified PAN:** Thay shortcut connection từ addition sang concatenation.

---

## 4. EXPERIMENTS

### 4.1. Ảnh hưởng của từng feature (Ablation Study)

**BoF trên classifier (CSPResNeXt-50):**

| Feature | Top-1 |
|---------|-------|
| Baseline | 77.2% |
| + CutMix | 77.6% |
| + Mosaic | 77.8% |
| + Label Smoothing | 77.5% |
| + MixUp | 77.0% (giảm!) |

**BoF trên detector:**

| Feature | mAP | mAP50 |
|---------|-----|-------|
| Baseline (SPP + PAN) | 38.0 | - |
| + Mosaic + SAT + GA | 39.3 | - |
| + CIoU | 39.6 | - |
| + Eliminate grid sensitivity | 39.8 | - |
| + Multiple anchors/GT | 40.3 | - |
| + CmBN + DropBlock + Cosine | 40.7 | - |
| **ALL (YOLOv4)** | **43.5** | **65.7** |

### 4.2. So sánh SOTA

| Phương pháp | mAP50-95 | FPS |
|------------|---------|-----|
| EfficientDet-D3 | 45.8 | 24 |
| YOLOv3 | 33.0 | 45 |
| **YOLOv4** | **43.5** | **~65** |

YOLOv4 nhanh hơn EfficientDet 2× và accuracy chỉ kém 2.3%.

---

## 5. CONCLUSION

YOLOv4 chạy 2× nhanh hơn EfficientDet với hiệu suất tương đương. Cải thiện YOLOv3 về AP và FPS 10% và 12% tương ứng. Chứng minh BoF và BoS có ảnh hưởng lớn và có thể train hiệu quả trên single GPU.

---

*Bản dịch chi tiết từ paper gốc arXiv:2004.10934.*
