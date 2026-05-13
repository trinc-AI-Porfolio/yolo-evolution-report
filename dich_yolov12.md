# DỊCH CHI TIẾT PAPER YOLOv12
**"YOLOv12: Attention-Centric Real-Time Object Detectors"**
**Tác giả:** Yunjie Tian, Qixiang Ye, David Doermann
**Năm:** 2025 | **Link:** https://arxiv.org/abs/2502.12524

---

## ABSTRACT

YOLO truyền thống dựa trên CNN. Attention mechanisms có global receptive field tốt hơn nhưng quá chậm cho real-time. Paper đề xuất **Area Attention (A²)** và **R-ELAN** để đưa attention vào YOLO mà giữ tốc độ real-time. YOLOv12-N đạt 40.6% mAP (vượt YOLO11-N 39.5% và YOLOv10-N 38.5%).

---

## 1. INTRODUCTION

CNN-based YOLO đã chạm trần hiệu suất. Attention mechanisms (Transformers) có ưu điểm: global receptive field, content-adaptive processing. Nhưng self-attention có complexity O((HW)²) → quá chậm.

Thách thức khi đưa attention vào YOLO:
1. **Tốc độ:** Self-attention quá chậm cho real-time
2. **Optimization:** Attention networks khó train hơn CNN
3. **Hardware:** Attention operations chưa được tối ưu trên GPU bằng Conv

---

## 2. METHOD

### 2.1. Area Attention (A²)

**Ý tưởng:** Thay vì global attention trên toàn bộ feature map, chia thành **n vùng bằng nhau** và tính attention **trong từng vùng riêng biệt**.

**Complexity:**
- Global self-attention: O((H×W)²)
- Area Attention (n vùng): O(H×W × (HW/n))
- Với n=4: giảm **4× complexity**

**3 cách chia vùng:**
1. Chia theo chiều ngang: n vùng nằm ngang
2. Chia theo chiều dọc: n vùng dọc
3. Chia theo cả 2: √n × √n grid

Thí nghiệm: Area Attention (n=4) chỉ mất **0.1% AP** so với global attention nhưng nhanh hơn rất nhiều.

### 2.2. R-ELAN (Residual ELAN)

**Vấn đề:** Khi thay CNN blocks bằng attention blocks trong ELAN → training **bất ổn**, accuracy giảm.

**Nguyên nhân:** Attention outputs có variance lớn ở early training → khi concatenate nhiều attention branches → tích lũy variance → gradient explode.

**Giải pháp R-ELAN:**
1. Thêm **residual shortcut** với **scale factor nhỏ (0.01)**
2. Scale factor nhỏ → output gần identity ở đầu → gradient ổn định
3. Dần dần model học scale factor phù hợp

**So sánh:**
- ELAN + Attention: Bất ổn, accuracy thấp ❌
- R-ELAN + Attention: Ổn định, accuracy cao ✅

### 2.3. Architectural Refinements

**FlashAttention:** Tối ưu memory cho attention computation → giảm memory footprint.

**Conv(3×3) + BatchNorm thay Linear + LayerNorm:**
- Linear + LN: Chuẩn cho Transformers nhưng chậm trên GPU (sequential operations)
- Conv + BN: Tận dụng hardware parallelism tốt hơn → nhanh hơn

**MLP ratio giảm:** 4.0 (Transformer chuẩn) → 1.2-2.0. Giảm computation trong FFN mà giữ accuracy.

**Bỏ Positional Encoding:** Thay bằng **7×7 depthwise conv** — implicit position information từ local convolution. Đơn giản hơn, hiệu quả tương đương.

---

## 3. EXPERIMENTS

### So sánh SOTA:

| Model | Params | GFLOPs | mAP50-95 |
|-------|--------|--------|---------|
| YOLOv8-N | 3.2M | 8.7 | 37.3% |
| YOLOv10-N | 2.3M | 6.7 | 38.5% |
| YOLO11-N | 2.6M | 6.5 | 39.5% |
| **YOLOv12-N** | **2.6M** | **6.5** | **40.6%** |
| YOLO11-S | 9.4M | 21.5 | 47.0% |
| **YOLOv12-S** | **9.3M** | **21.4** | **48.0%** |
| YOLO11-L | 25.3M | 87.6 | 53.4% |
| **YOLOv12-L** | **26.4M** | **88.9** | **53.7%** |
| **YOLOv12-X** | **59.1M** | **199.0** | **55.2%** |

### Ablation: Area Attention

| Attention Type | mAP | Latency |
|---------------|-----|---------|
| Global Self-Attention | 40.7% | Chậm |
| Window Attention (SWIN) | 40.3% | Trung bình |
| **Area Attention (n=4)** | **40.6%** | **Nhanh** |

Area Attention chỉ kém global 0.1% mAP nhưng nhanh hơn rất nhiều.

### Ablation: R-ELAN

| Architecture | mAP | Stable? |
|-------------|-----|---------|
| ELAN + Attention | 39.1% | ❌ |
| ELAN + Attention + Residual | 39.8% | ⚠️ |
| **R-ELAN + Attention (scale=0.01)** | **40.6%** | **✅** |

---

## 4. CONCLUSION

YOLOv12 là bước đầu đưa attention-centric design vào YOLO thành công. Area Attention giải quyết vấn đề tốc độ, R-ELAN giải quyết vấn đề ổn định training. Mở ra hướng mới: YOLO không còn chỉ là CNN.

---

*Bản dịch chi tiết từ paper gốc arXiv:2502.12524.*
