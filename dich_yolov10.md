# DỊCH CHI TIẾT PAPER YOLOv10
**"YOLOv10: Real-Time End-to-End Object Detection"**
**Tác giả:** Ao Wang et al. (Tsinghua University)
**Năm:** 2024 | **Link:** https://arxiv.org/abs/2405.14458

---

## ABSTRACT

YOLOs dựa vào NMS khi post-processing → làm chậm inference, không end-to-end. Paper đề xuất **Consistent Dual Assignments** để loại bỏ NMS, kết hợp với **holistic efficiency-accuracy design** → YOLOv10 đạt SOTA performance-latency tradeoff. v10-S: 46.3% mAP, 2.49ms latency (T4 GPU). v10-X: 54.4% mAP với 29.5M params (ít hơn YOLOv8-X 2.3×).

---

## 1. INTRODUCTION

Vấn đề NMS:
1. **Tốn thời gian:** Cần tune confidence threshold, IoU threshold
2. **Không differentiable:** Không thể tối ưu end-to-end
3. **Latency phụ thuộc data:** Thời gian NMS thay đổi theo số detections

DETR loại bỏ NMS bằng one-to-one matching nhưng quá chậm. YOLOv10 đề xuất cách loại bỏ NMS mà giữ tốc độ YOLO.

---

## 2. METHODOLOGY

### 2.1. Consistent Dual Assignments for NMS-Free Training

**Vấn đề core:** 
- **One-to-many (o2m):** 1 GT object → gán nhiều predictions → training loss phong phú → train tốt
- **One-to-one (o2o):** 1 GT object → gán 1 prediction → inference không cần NMS → deploy tốt
- Nhưng o2o training kém hơn o2m vì ít supervision signals

**Giải pháp: Dual Heads**
- Training: Dùng **cả 2 heads** — o2m head tạo dense supervision, o2o head học matching
- Inference: Chỉ dùng **o2o head** → không cần NMS

**Consistent Matching Metric:**
Đảm bảo o2m và o2o nhất quán — prediction tốt nhất theo o2m cũng là prediction được chọn bởi o2o:
> m(α, β) = s^α × IoU^β
- s = classification score, IoU = localization quality
- Dùng cùng metric cho cả o2m và o2o → consistent

### 2.2. Holistic Efficiency-Accuracy Driven Design

**A. Lightweight Classification Head:**
- Quan sát: Classification head tốn nhiều computation hơn regression head nhưng ít ảnh hưởng accuracy
- Giải pháp: Dùng **depthwise separable convolution** cho cls head, giữ regular conv cho reg head
- Giảm significant overhead với minimal accuracy loss

**B. Spatial-Channel Decoupled Downsampling:**
- Thông thường: 3×3 conv stride 2 → spatial reduction + channel increase đồng thời → expensive
- Đề xuất: Tách thành pointwise conv (tăng channels) → depthwise conv (giảm spatial) → rẻ hơn

**C. Rank-Guided Block Design:**
- Phân tích **intrinsic rank** (số singular values lớn) của mỗi stage
- Stage có rank thấp = redundancy cao → dùng block nhẹ hơn (CIB - Compact Inverted Block)
- Stage có rank cao = cần complexity → giữ block nặng
- Tự động quyết định block nào dùng ở đâu

**D. Large-Kernel Convolution:**
- Dùng **7×7 depthwise conv** ở stages sâu
- Tăng receptive field → capture long-range dependencies
- Chỉ ở stages sâu (stages nông đã có đủ local features)

**E. Partial Self-Attention (PSA):**
- Self-attention tốt nhưng expensive
- PSA: Chia channels thành 2 phần, chỉ **1 phần qua self-attention**, phần còn lại bypass
- Giảm cost 50% nhưng giữ phần lớn benefit
- Chỉ dùng ở stages cuối (resolution nhỏ → attention rẻ hơn)

---

## 3. EXPERIMENTS

### So sánh SOTA:

| Model | Params | Latency | mAP50-95 |
|-------|--------|---------|---------|
| YOLOv8-N | 3.2M | 1.77ms | 37.3% |
| **YOLOv10-N** | **2.3M** | **1.84ms** | **38.5%** |
| YOLOv8-S | 11.2M | 2.33ms | 44.9% |
| **YOLOv10-S** | **7.2M** | **2.49ms** | **46.3%** |
| YOLOv8-X | 68.2M | 7.61ms | 53.9% |
| **YOLOv10-X** | **29.5M** | **10.70ms** | **54.4%** |
| RT-DETR-R50 | 42M | 9.35ms | 53.1% |

YOLOv10: Luôn ít params hơn, accuracy cao hơn. v10-X chỉ 29.5M params vs v8-X 68.2M (2.3× ít hơn).

### Ablation: NMS-Free

| Method | mAP | Latency |
|--------|-----|---------|
| YOLOv8-S + NMS | 44.9% | 4.70ms |
| o2o only training | 43.8% | 2.33ms |
| **Dual Assignment** | **44.3%** | **2.33ms** |

Dual Assignment gần bằng NMS-based accuracy nhưng **nhanh hơn 2× (loại bỏ NMS latency)**.

---

*Bản dịch chi tiết từ paper gốc arXiv:2405.14458.*
