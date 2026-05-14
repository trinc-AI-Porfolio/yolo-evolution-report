# YOLO26 — TÀI LIỆU KỸ THUẬT CHI TIẾT (Không có paper chính thức)
**Tác giả:** Glenn Jocher / Ultralytics
**Năm:** 2026 (release 01/2026)
**Nguồn:** GitHub (github.com/ultralytics/ultralytics), Ultralytics Blog, Release Notes
**Framework:** PyTorch (package `ultralytics`)

---

## BỐI CẢNH

YOLO26 là phiên bản mới nhất, tập trung vào **Edge-First Design** — tối ưu cho thiết bị edge (mobile, camera nhúng, drone) mà vẫn đạt mAP cao nhất YOLO family. Đánh dấu bước chuyển: YOLO không chỉ cho server GPU mà phải chạy tốt trên mọi phần cứng.

**Nguồn thông tin:** Ultralytics blog, GitHub release notes, community benchmarks, source code analysis. Không có peer-reviewed publication.

---

## KIẾN TRÚC

### Thay đổi lớn nhất: BỎ DFL

**Distribution Focal Loss (DFL)** được giới thiệu ở v8 — dự đoán phân phối xác suất cho box coordinates (16 bins/coordinate). Tuy accuracy tốt nhưng:
- Model graph phức tạp → export khó
- Thêm post-processing (weighted sum) → chậm trên edge
- Không phải mọi hardware hỗ trợ DFL operations hiệu quả

YOLO26 **bỏ DFL hoàn toàn** → quay về dự đoán trực tiếp 4 coordinates. Bù lại bằng training techniques mạnh hơn (MuSGD, ProgLoss). Kết quả: model graph đơn giản hơn, export dễ hơn, inference nhanh hơn trên edge.

### Backbone & Neck
- Kế thừa C3K2 + C2PSA từ YOLO11
- Tối ưu thêm cho edge inference: giảm redundant operations
- NMS-free native (không cần NMS)

### Model Sizes

| Model | Params | mAP50-95 | So với YOLO11 | CPU Speed |
|-------|--------|---------|-------------|-----------|
| 26n | ~2.5M | ~40.5% | 11n: 39.5% (+1%) | **Nhanh hơn 43%** |
| 26s | ~9M | ~47.5% | 11s: 47.0% (+0.5%) | Nhanh hơn ~30% |
| 26m | ~20M | ~52.0% | 11m: 51.5% (+0.5%) | Nhanh hơn ~25% |
| 26l | ~25M | ~54.5% | 11l: 53.4% (+1.1%) | Nhanh hơn ~20% |
| 26x | ~60M | ~57.2% | 11x: 54.7% (+2.5%) | Nhanh hơn ~15% |

*Nguồn: Ultralytics blog, community benchmarks. Dấu ~ vì không có paper chính thức xác nhận.*

**Nhận xét:** YOLO26-X (~57.2%) vượt kỷ lục mAP của v7-E6E (56.8%) — lần đầu tiên sau 4 năm. Đặc biệt nhanh hơn trên CPU → phù hợp edge deployment.

---

## PHƯƠNG PHÁP MỚI

### 1. MuSGD (Muon SGD)
- Kết hợp **SGD truyền thống** với **Muon optimizer** (kỹ thuật từ Large Language Models)
- Muon: Momentum-based update với orthogonal projection → gradient ổn định hơn
- Kết quả: Training hội tụ nhanh hơn, ít cần fine-tune learning rate
- Cross-pollination: Kỹ thuật từ LLM training áp dụng cho CV — xu hướng mới

### 2. STAL (Small-Target-Aware Label Assignment)
- Vấn đề: Label assignment truyền thống (TAL) bias toward vật lớn — vật lớn có IoU cao hơn tự nhiên
- STAL: Điều chỉnh assignment metric cho vật nhỏ (< 8 pixels)
- Tăng weight cho small targets trong matching → nhiều positive samples hơn cho vật nhỏ
- Kết quả: Phát hiện vật nhỏ cải thiện đáng kể — quan trọng cho drone, surveillance

### 3. ProgLoss (Progressive Loss Balancing)
- Vấn đề: Tỷ lệ giữa box loss, cls loss, dfl loss cố định suốt training → không tối ưu
- ProgLoss: **Tự động cân bằng** loss weights theo training progress
- Giai đoạn đầu: Ưu tiên box loss (học localize trước)
- Giai đoạn sau: Tăng cls loss (refine classification)
- Tự điều chỉnh dựa trên gradient magnitudes

### 4. NMS-Free Native
- Kế thừa dual assignment từ v10/v11
- Tích hợp sâu hơn vào kiến trúc (không phải add-on)
- True end-to-end detection

---

## MULTI-TASK

Hỗ trợ 5 tác vụ (giảm 1 so với v8/v11):
- Detection ✅
- Instance Segmentation ✅
- Classification ✅
- Pose Estimation ✅
- OBB (Oriented Bounding Box) ✅
- Tracking ✅ (qua API)

---

## EDGE-FIRST DESIGN PHILOSOPHY

YOLO26 được thiết kế từ đầu cho edge:
1. **Bỏ DFL** → model graph đơn giản → ONNX/TensorRT export sạch
2. **Giảm post-processing** → ít custom ops → tương thích nhiều hardware hơn
3. **CPU optimization** → nhanh hơn 43% trên CPU vs YOLO11
4. **Quantization-friendly** → INT8 mất ít accuracy hơn

---

## ĐÓNG GÓP CHÍNH
1. **Bỏ DFL:** Đơn giản hóa, edge-friendly
2. **MuSGD:** Cross-pollination LLM → CV
3. **STAL:** Giải quyết vật nhỏ
4. **ProgLoss:** Adaptive loss balancing
5. **mAP cao nhất:** ~57.2% — vượt kỷ lục v7
6. **Edge-first:** Nhanh hơn 43% CPU

## HẠN CHẾ
1. **Quá mới** — chưa có đủ community testing
2. Không paper → claims chưa peer-reviewed
3. Thông số benchmark có thể thay đổi qua updates
4. STAL và ProgLoss chưa có phân tích lý thuyết chi tiết

---

## SO SÁNH VỚI ĐỜI TRƯỚC

| Tiêu chí | YOLO11 | YOLO26 | Thay đổi |
|----------|--------|--------|---------|
| mAP (X) | 54.7% | ~57.2% | +2.5% |
| DFL | Có | **Bỏ** | Đơn giản hơn |
| CPU speed | Baseline | +43% | Cải thiện lớn |
| Optimizer | SGD | **MuSGD** | Từ LLM |
| Vật nhỏ | Bình thường | **STAL** | Cải thiện |
| Loss | Cố định | **ProgLoss** | Adaptive |

---

*Tổng hợp từ: Ultralytics Blog, GitHub, Release Notes. Không phải peer-reviewed paper. Thông số có thể thay đổi qua updates.*
