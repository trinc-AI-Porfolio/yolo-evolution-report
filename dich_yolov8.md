# YOLOv8 — TÀI LIỆU KỸ THUẬT CHI TIẾT (Không có paper chính thức)
**Tác giả:** Glenn Jocher / Ultralytics
**Năm:** 2023 (release 01/2023)
**Nguồn:** GitHub (github.com/ultralytics/ultralytics), Ultralytics Docs, Blog posts
**Framework:** PyTorch (package `ultralytics`)

---

## BỐI CẢNH

YOLOv8 là sự chuyển đổi lớn nhất của Ultralytics — từ "model" sang "platform". Không chỉ là detector mà là **framework multi-task thống nhất** hỗ trợ 6 tác vụ. Đánh dấu bước chuyển từ anchor-based sang anchor-free trong hệ sinh thái Ultralytics.

**Nguồn thông tin:** Ultralytics blog, GitHub source code analysis, community benchmarks, documentation chính thức. Không có peer-reviewed publication.

---

## KIẾN TRÚC

### Backbone: CSPDarknet (Modified với C2f)
- **C2f module** (Cross Stage Partial v2 with 2 convolutions): Kế thừa ý tưởng C3 (v5) + ELAN (v7)
- C2f khác C3: Cho phép **tất cả** bottleneck outputs concat lại (thay vì chỉ output cuối) → gradient flow phong phú hơn, nhiều features hơn
- Cấu trúc: Conv 1×1 (split) → N × Bottleneck → Conv 1×1 (merge), với skip connections từ mọi bottleneck

### Neck: PAN-FPN (modified)
- Tương tự v5 nhưng dùng C2f thay C3
- Top-down (FPN) + Bottom-up (PAN) feature aggregation

### Head: **Decoupled Anchor-Free Head** — Thay đổi lớn nhất
**3 thay đổi quan trọng so với v5:**

**1. Anchor-Free:**
- Bỏ anchor boxes hoàn toàn
- Dự đoán trực tiếp center point + 4 distances (top, bottom, left, right)
- Không cần AutoAnchor, đơn giản hơn

**2. Decoupled Head:**
- Tách classification và regression thành **2 branches riêng biệt**
- v5 dùng coupled head: 1 branch dự đoán cả class + box → 2 tasks compete for capacity
- v8: Classification branch (cls) + Regression branch (reg) → mỗi branch tối ưu cho task riêng

**3. Bỏ Objectness Branch:**
- v5: 3 outputs — objectness + classification + regression
- v8: 2 outputs — classification + regression
- Lý do: Với anchor-free + TAL, objectness trở nên redundant

### Distribution Focal Loss (DFL)
- Thay vì dự đoán 1 giá trị cho mỗi coordinate (top/bottom/left/right)
- DFL dự đoán **phân phối xác suất** trên nhiều bins (thường 16 bins)
- Model biểu diễn uncertainty: "cạnh phải có thể ở pixel 10, 11, hoặc 12"
- Kết quả cuối: Weighted sum theo phân phối → localization chính xác hơn

### Label Assignment: TAL (Task-Aligned Learning)
- Kế thừa từ YOLOv6/TOOD
- Alignment metric: m = s^α × u^β (s = cls score, u = IoU)
- Dynamic assignment: Top-k predictions theo metric → positive samples
- Đảm bảo cls score cao ↔ localization tốt → consistent predictions

### 6 Model Sizes

| Model | Params | FLOPs | mAP50-95 |
|-------|--------|-------|---------|
| v8n | 3.2M | 8.7G | 37.3% |
| v8s | 11.2M | 28.6G | 44.9% |
| v8m | 25.9M | 78.9G | 50.2% |
| v8l | 47.3M | 165.2G | 52.9% |
| v8x | 68.2M | 257.8G | 53.9% |

*Nguồn: Ultralytics docs, COCO val2017, input 640.*

---

## MULTI-TASK — Điểm mạnh lớn nhất

YOLOv8 hỗ trợ **6 tác vụ** trong 1 framework duy nhất:

| Task | Mô tả | API |
|------|--------|-----|
| **Detection** | Bounding box + class | `model.predict()` |
| **Segmentation** | Instance segmentation (mask pixel-level) | `YOLO('yolov8n-seg.pt')` |
| **Classification** | Image classification | `YOLO('yolov8n-cls.pt')` |
| **Pose Estimation** | Keypoint detection (17 body joints) | `YOLO('yolov8n-pose.pt')` |
| **OBB** | Oriented/Rotated bounding boxes | `YOLO('yolov8n-obb.pt')` |
| **Tracking** | Multi-object tracking (ByteTrack, BoT-SORT) | `model.track()` |

Tất cả dùng cùng 1 API, cùng 1 CLI, cùng 1 pipeline training.

---

## TRAINING DETAILS

### Loss Function
- **Box loss:** CIoU + DFL (Distribution Focal Loss)
- **Cls loss:** BCE (Binary Cross Entropy) với variant of Focal Loss
- **DFL loss:** Cross entropy trên distribution bins

### Augmentation
Kế thừa v5: Mosaic, Copy-Paste, MixUp, HSV, Affine, Flip
- **Mosaic tắt ở 10 epochs cuối** — cho model ổn định cuối training
- Close-mosaic strategy cải thiện 0.5-1% mAP

### Training Config
- Optimizer: SGD (mặc định) hoặc AdamW
- 500 epochs mặc định (tăng từ 300 của v5)
- AMP (Automatic Mixed Precision) training
- EMA (Exponential Moving Average) model

---

## ĐÓNG GÓP CHÍNH
1. **Anchor-free + Decoupled head:** Hiện đại hóa YOLO architecture
2. **DFL:** Uncertainty-aware localization
3. **Multi-task 6 tác vụ:** Từ "detector" thành "platform"
4. **Unified API:** 1 dòng code cho mọi task
5. **Close-mosaic:** Cải thiện training stability

## HẠN CHẾ
1. Không paper → không peer review
2. mAP (53.9%) thấp hơn v7 (56.8%) ở cùng thời điểm
3. Vẫn cần NMS (chưa end-to-end)
4. DFL tăng complexity cho export/deploy

---

*Tổng hợp từ: Ultralytics GitHub, Blog, Docs, source code analysis.*
