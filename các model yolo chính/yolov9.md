# DỊCH CHI TIẾT PAPER YOLOv9
**"YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information"**
**Tác giả:** Chien-Yao Wang, I-Hau Yeh, Hong-Yuan Mark Liao
**Năm:** 2024 | **Link:** https://arxiv.org/abs/2402.13616

---

## ABSTRACT

Vấn đề hiện tại của deep learning: Khi dữ liệu đi qua nhiều layers, **thông tin bị mất dần** (information bottleneck). Điều này khiến gradient không đáng tin cậy → model không học được đầy đủ. Paper đề xuất **PGI (Programmable Gradient Information)** và **GELAN (Generalized ELAN)** để giải quyết. YOLOv9 đạt SOTA trên COCO: v9-C 53.0% mAP (25.5M params), v9-E 55.6% mAP (58.1M params) — ít params hơn YOLOv8 mà accuracy cao hơn.

---

## 1. INTRODUCTION

Deep learning đã thành công lớn nhờ khả năng trích xuất features tự động. Tuy nhiên, khi network sâu hơn, vấn đề **information bottleneck** nảy sinh: layers sâu mất thông tin quan trọng từ input gốc.

Các giải pháp trước: Reversible architectures (giữ toàn bộ thông tin nhưng nặng), deep supervision (thêm loss ở giữa), knowledge distillation (cần model lớn hơn).

YOLOv9 đề xuất cách tiếp cận mới: dùng **auxiliary reversible branch** tạo gradient đáng tin cậy cho main branch, không tăng inference cost.

---

## 2. RELATED WORK

### 2.1. Information Bottleneck Principle

Theo lý thuyết Information Bottleneck, khi data X đi qua transformation fθ:

> I(X, X) ≥ I(X, f₁(X)) ≥ I(X, f₂(f₁(X))) ≥ ...

Mutual information **giảm đơn điệu** qua mỗi layer. Nghĩa là layers sâu biết ít hơn về input gốc → gradient dựa trên thông tin không đầy đủ → model không học được tối ưu.

### 2.2. Reversible Functions

Reversible function: có thể khôi phục input từ output (r(y) = x). Nếu toàn bộ network reversible → không mất thông tin. Nhưng reversible architectures thường nặng và hạn chế kiến trúc.

---

## 3. METHOD

### 3.1. Programmable Gradient Information (PGI)

PGI gồm 3 thành phần:

**1. Main Branch:**
- Network chính dùng cho inference
- Kiến trúc tùy chọn (không bị ràng buộc reversible)
- Giữ nguyên khi deploy

**2. Auxiliary Reversible Branch:**
- Mạng phụ reversible (giữ complete information)
- Tạo reliable gradient signals
- **Chỉ dùng khi training, bỏ khi inference** → zero inference cost
- Giải quyết: deep supervision truyền thống chỉ thêm loss ở giữa nhưng gradient vẫn dựa trên features đã mất thông tin. PGI dùng reversible branch nên gradient dựa trên **complete information**

**3. Multi-level Auxiliary Information:**
- Kết nối auxiliary branch với main branch ở **nhiều tầng**
- Tránh vấn đề "broken gradient flow" khi main branch rất sâu
- Gradient được inject trực tiếp vào các intermediate layers

**Tại sao PGI tốt hơn deep supervision?**
- Deep supervision: Thêm loss ở giữa, nhưng features ở giữa đã mất thông tin → gradient không đáng tin
- PGI: Auxiliary reversible branch **giữ full information** → gradient luôn đáng tin cậy
- PGI đặc biệt hiệu quả cho **lightweight models** (cải thiện >2% AP) vì models nhỏ mất thông tin nhiều hơn

### 3.2. GELAN (Generalized ELAN)

GELAN tổng quát hóa CSPNet + ELAN:
- Cho phép dùng **bất kỳ computational block** nào bên trong (Conv, Bottleneck, Transformer...)
- Thay block mà không cần redesign kiến trúc
- Linh hoạt hơn nhiều so với ELAN cố định

Thí nghiệm: Thay RepConv bằng Res-Block, Dark-Block, CSP-Block → GELAN đều hoạt động tốt → chứng minh tính tổng quát.

---

## 4. EXPERIMENTS

### 4.1. So sánh SOTA trên COCO

| Model | Params | FLOPs | mAP50-95 |
|-------|--------|-------|---------|
| YOLOv8-S | 11.2M | 28.6G | 44.9% |
| **YOLOv9-S** | **7.2M** | **26.7G** | **46.8%** |
| YOLOv8-M | 25.9M | 78.9G | 50.2% |
| **YOLOv9-C** | **25.5M** | **102.8G** | **53.0%** |
| YOLOv8-X | 68.2M | 257.8G | 53.9% |
| **YOLOv9-E** | **58.1M** | **192.5G** | **55.6%** |

**Nhận xét:** YOLOv9 luôn ít params hơn YOLOv8 mà accuracy cao hơn ở mọi scale.

### 4.2. Ablation Study — PGI

| Method | mAP (light) | mAP (medium) |
|--------|------------|-------------|
| Baseline | 37.3% | 49.1% |
| + Deep Supervision | 38.2% | 49.3% |
| + **PGI** | **39.8%** | **49.6%** |

PGI cải thiện lightweight model **+2.5%** vs chỉ +0.9% cho medium model → PGI đặc biệt quan trọng cho models nhỏ.

### 4.3. GELAN Flexibility

Thay computational block trong GELAN:
- ELAN + CSPBlock: 52.3% mAP
- ELAN + DarkBlock: 52.4% mAP  
- **GELAN** (mixed): **53.0% mAP** → tốt nhất khi kết hợp nhiều loại blocks

---

## 5. CONCLUSION

YOLOv9 giải quyết vấn đề information bottleneck bằng PGI — đóng góp lý thuyết sâu sắc nhất trong lịch sử YOLO. GELAN cung cấp kiến trúc linh hoạt tổng quát. Kết quả: ít params, accuracy cao hơn tất cả YOLO trước đó.

---

*Bản dịch chi tiết từ paper gốc arXiv:2402.13616.*
