# YOLO11 — TÀI LIỆU KỸ THUẬT CHI TIẾT (Không có paper chính thức)
**Tác giả:** Glenn Jocher / Ultralytics
**Năm:** 2024 (release 09/2024)
**Nguồn:** GitHub (github.com/ultralytics/ultralytics), Ultralytics Blog, Docs
**Framework:** PyTorch (package `ultralytics`)

---

## BỐI CẢNH

YOLO11 (không có "v" — Ultralytics đổi naming convention) là bản nâng cấp toàn diện của v8. Tập trung vào **efficiency** — cùng hoặc ít params hơn v8 nhưng accuracy cao hơn ở mọi scale. Kế thừa NMS-free option từ v10.

**Lưu ý naming:** Ultralytics bỏ "v" → YOLO11 thay vì YOLOv11. Lý do: YOLO đã trở thành brand, không chỉ là version number.

**Nguồn thông tin:** Ultralytics blog announcement, GitHub source code, documentation, community benchmarks.

---

## KIẾN TRÚC

### Backbone: CSPDarknet với C3K2

**C3K2 (CSP Bottleneck with 2 Kernel sizes):**
- Kế thừa C2f (v8) nhưng dùng **2 kernel sizes: 3×3 và 5×5**
- Kernel 3×3: Capture local features (chi tiết nhỏ)
- Kernel 5×5: Capture wider context (vùng rộng hơn)
- Multi-scale feature extraction **trong cùng 1 block** → phong phú hơn C2f (chỉ dùng 3×3)

### Neck: PAN-FPN với C2PSA

**C2PSA (Cross Stage Partial Spatial Attention):**
- Thêm **spatial attention** vào CSP structure
- Cách hoạt động:
  1. Features đi qua CSP split
  2. Một nhánh qua **spatial attention module** — học "vùng nào quan trọng"
  3. Attention map highlight vùng có vật thể, suppress background
  4. Concat với nhánh bypass → output
- So với C2f (v8): C2PSA **biết focus vào đâu**, C2f chỉ trích xuất features đều

### Head: Anchor-free, Decoupled (giống v8)
- Kế thừa kiến trúc head từ v8: Decoupled cls/reg, anchor-free, DFL
- Thêm NMS-free option (kế thừa từ v10 dual assignments)

### Model Sizes

| Model | Params | FLOPs | mAP50-95 | So với v8 |
|-------|--------|-------|---------|----------|
| 11n | 2.6M | 6.5G | 39.5% | v8n: 37.3% (**+2.2%**) |
| 11s | 9.4M | 21.5G | 47.0% | v8s: 44.9% (**+2.1%**) |
| 11m | 20.1M | 68.0G | 51.5% | v8m: 50.2% (**+1.3%**) |
| 11l | 25.3M | 87.6G | 53.4% | v8l: 52.9% (**+0.5%**) |
| 11x | 56.9M | 194.9G | 54.7% | v8x: 53.9% (**+0.8%**) |

*Nguồn: Ultralytics docs, COCO val2017, input 640.*

**Nhận xét:** YOLO11 cải thiện ở MỌI scale, đặc biệt mạnh ở Nano/Small (+2%). Params tương đương hoặc ít hơn v8.

---

## MULTI-TASK

Kế thừa đầy đủ 6 tác vụ từ v8:
Detection, Segmentation, Classification, Pose, OBB, Tracking

Thêm cải tiến:
- **NMS-free inference** (optional) → true end-to-end
- Speed improvements cho segmentation và pose tasks
- Better support cho edge deployment

---

## TRAINING DETAILS

### Cải tiến so với v8
- Training recipe tối ưu hơn (cùng epochs nhưng converge tốt hơn)
- Close-mosaic strategy cải tiến
- Better augmentation scheduling

### Loss Function
Giống v8: CIoU + DFL + BCE
- Có option NMS-free loss (dual assignment từ v10)

---

## ĐÓNG GÓP CHÍNH
1. **C3K2:** Multi-scale kernels trong 1 block → efficiency cao hơn
2. **C2PSA:** Spatial attention giúp focus vào vùng quan trọng
3. **Efficiency:** Ít params hơn v8 mà accuracy cao hơn ở mọi scale
4. **NMS-free option:** Kế thừa từ v10
5. **Naming mới:** Bỏ "v", YOLO trở thành brand

## HẠN CHẾ
1. Cải tiến incremental, không đột phá
2. Không paper → không rõ chi tiết ablation
3. C2PSA chưa được phân tích lý thuyết sâu
4. Vẫn CNN-centric (v12 sau đó mới chuyển attention)

---

*Tổng hợp từ: Ultralytics Blog, GitHub, Docs, source code.*
