# TỔNG HỢP TOÀN BỘ CÁC ĐỜI YOLO (You Only Look Once)
*Từ YOLOv1 (2015) đến YOLO26 (2026) + Các biến thể*

---

## 📖 YOLO là gì?

**YOLO = "You Only Look Once"** — mô hình AI phát hiện vật thể trong ảnh/video chỉ bằng 1 lần nhìn.

**Trước YOLO** (R-CNN): Quét từng vùng ảnh → cực chậm (vài giây/ảnh)  
**YOLO**: Nhìn toàn bộ ảnh 1 lần → cực nhanh (hàng chục ảnh/giây, real-time)

### Thuật ngữ cơ bản

| Thuật ngữ | Giải thích | Ví dụ |
|-----------|-----------|-------|
| **Bounding Box** | Khung chữ nhật bao quanh vật thể | Khung quanh khuôn mặt khi mở camera |
| **mAP** | Điểm chính xác (0-100%) | 50% = đúng ~nửa số vật |
| **FPS** | Số ảnh xử lý/giây | 30 FPS = video mượt |
| **Backbone** | Phần trích xuất đặc trưng | "Bộ não" nhận biết hình dạng |
| **Neck** | Kết nối backbone với head | "Cổ" nối não với thân |
| **Head** | Phần đưa ra kết quả | "Miệng" nói đáp án |
| **Anchor Box** | Khung mẫu có sẵn | Template hình chữ nhật |
| **NMS** | Lọc box trùng lặp | 5 box cùng 1 vật → giữ 1 tốt nhất |
| **IoU** | Đo trùng lặp 2 box (0-1) | 1 = trùng hoàn hảo |

---

# PHẦN 1: DÒNG CHÍNH (v1 → v12 → YOLOE → YOLO26)

---

## 1. YOLOv1 (2015) — Khởi nguồn
**Tác giả:** Joseph Redmon | **Backbone:** 24-layer CNN + 2 FC | **Grid:** 7×7

Chia ảnh thành lưới 7×7 (49 ô). Mỗi ô dự đoán 2 box + 20 classes.

> **Ví dụ:** Liếc mắt 1 cái thấy quả táo ở kệ nào trong siêu thị, thay vì đi từng kệ kiểm tra.

| Chỉ số | Giá trị |
|--------|---------|
| mAP | 63.4% (VOC) |
| FPS | 45 |

**✅ Ưu:** Cực nhanh, real-time đầu tiên, hiểu context toàn ảnh  
**❌ Nhược:** Vật nhỏ kém (grid thô), localization sai, mỗi ô chỉ 1 class  
> *Tóm: Sinh viên mới ra trường — nhanh nhưng thiếu kinh nghiệm*

---

## 2. YOLOv2 / YOLO9000 (2016) — Better, Faster, Stronger
**Tác giả:** Joseph Redmon | **Backbone:** Darknet-19 | **Grid:** 13×13

**Cải tiến:** Anchor boxes (5 anchors/ô), Batch Norm (+2% mAP), Multi-scale training, bỏ FC layers. YOLO9000 phát hiện 9000+ categories.

> **Ví dụ Anchor:** Như có sẵn khung oval để vẽ chân dung → chỉ cần chỉnh lại cho vừa.

| Chỉ số | Giá trị |
|--------|---------|
| mAP | 78.6% (VOC) |
| FPS | 40 |

**✅ Ưu:** Chính xác hơn v1, anchor cải thiện recall, linh hoạt input size  
**❌ Nhược:** Chỉ 1 scale, 1 box = 1 class (Softmax)  
> *Tóm: Đã đi làm 1 năm — kỹ năng tốt hơn nhưng vẫn có giới hạn*

---

## 3. YOLOv3 (2018) — Multi-Scale Detection
**Tác giả:** Joseph Redmon *(bản cuối cùng của ông)* | **Backbone:** Darknet-53 (residual)

**Đột phá:** Dự đoán ở **3 scales** (13×13, 26×26, 52×52) — bắt vật nhỏ tốt hơn. Multi-label (sigmoid thay softmax).

> **Ví dụ:** 3 ống kính zoom khác nhau nhìn cùng lúc: zoom xa (vật lớn), trung bình, zoom gần (vật nhỏ).

| Chỉ số | Giá trị |
|--------|---------|
| mAP50-95 | 33.0% (COCO) |
| FPS | ~30 |

**✅ Ưu:** Vật nhỏ tốt hơn, backbone mạnh (residual), multi-label  
**❌ Nhược:** Vật cỡ trung bình chưa tối ưu, anchor cố định  
> *Tóm: Nhân viên senior — nhìn vấn đề từ nhiều góc độ*

---

## 4. YOLOv4 (2020) — Tập đại thành
**Tác giả:** Alexei Bochkovskiy | **Backbone:** CSPDarknet53 | **Neck:** SPP + PANet

Gom mọi kỹ thuật hay nhất:
- **Bag of Freebies** (không tăng inference cost): Mosaic, CutMix, CIoU Loss, Label Smoothing
- **Bag of Specials** (tăng nhẹ cost): Mish activation, SPP, SAM, DIoU-NMS

> **Ví dụ:** Đầu bếp gom tất cả công thức hay nhất thế giới vào 1 cuốn sách.

| Chỉ số | Giá trị |
|--------|---------|
| mAP50-95 | 43.5% (COCO) |
| FPS | ~65 (V100) |

**✅ Ưu:** Accuracy nhảy vọt (+10.5%), train được 1 GPU, CIoU Loss  
**❌ Nhược:** Config phức tạp, Darknet (C) khó customize  
> *Tóm: Tiến sĩ tổng hợp — gom mọi tri thức thành hệ thống*

---

## 5. YOLOv5 (2020) — Dễ dùng nhất
**Tác giả:** Ultralytics | **Framework:** PyTorch | ⚠️ Không có paper

Chuyển từ Darknet C → PyTorch. 5 kích cỡ (n/s/m/l/x). AutoAnchor, export 10+ formats.

```python
from ultralytics import YOLO
model = YOLO('yolov5s.pt')
results = model('anh.jpg')  # 3 dòng code!
```

| Model | Params | mAP50-95 |
|-------|--------|----------|
| v5n | 1.9M | 28.0% |
| v5s | 7.2M | 37.4% |
| v5x | 86.7M | 50.7% |

**✅ Ưu:** Dễ nhất (pip install), PyTorch, community khổng lồ, export đa nền tảng  
**❌ Nhược:** Không paper, tên gây tranh cãi, anchor-based  
> *Tóm: iPhone — không mạnh nhất nhưng ai cũng dùng được*

---

## 6. YOLOv6 (2022) — Industrial
**Tác giả:** Meituan (美团) | **Anchor-free** | Decoupled head | Re-parameterization

> **Re-param:** Train phức tạp, inference đơn giản. Như ôn thi 5 cuốn sách, vào phòng thi chỉ cần 1 tờ tóm tắt.

| Chỉ số | Giá trị |
|--------|---------|
| mAP | 52.5% (L) |
| FPS | 520 (S, TensorRT) |

**✅ Ưu:** Anchor-free, cực nhanh (520 FPS!), tối ưu INT8  
**❌ Nhược:** Community nhỏ, ecosystem hạn chế  
> *Tóm: Xe tải chuyên dụng — mạnh trong công nghiệp*

---

## 7. YOLOv7 (2022) — SOTA Accuracy
**Tác giả:** Chien-Yao Wang | **E-ELAN** backbone | Compound scaling | Re-parameterization

| Chỉ số | Giá trị |
|--------|---------|
| mAP50-95 | 56.8% (E6E) |
| FPS | 161 (V100) |

**✅ Ưu:** SOTA accuracy 2022, E-ELAN tối ưu gradient  
**❌ Nhược:** Code phức tạp, anchor-based, ecosystem hạn chế  
> *Tóm: Vận động viên đỉnh cao — thành tích tốt nhất nhưng cần HLV giỏi*

---

## 8. YOLOv8 (2023) — Đa năng nhất
**Tác giả:** Ultralytics | **Anchor-free** | **Decoupled head** | C2f module

Multi-task: Detection + Segmentation + Classification + Pose + OBB + Tracking

| Model | Params | mAP50-95 |
|-------|--------|----------|
| v8n | 3.2M | 37.3% |
| v8s | 11.2M | 44.9% |
| v8x | 68.2M | 53.9% |

**✅ Ưu:** Multi-task tốt nhất, anchor-free, API thống nhất, community lớn nhất  
**❌ Nhược:** Không paper, vẫn cần NMS  
> *Tóm: Dao Thụy Sĩ — làm được MỌI THỨ*

---

## 9. YOLOv9 (2024) — Chống mất thông tin
**Tác giả:** Chien-Yao Wang | **PGI** (Programmable Gradient Information) | **GELAN**

> **PGI:** Nhánh phụ giữ thông tin gốc, như gửi SMS đối chiếu ngoài truyền miệng. Chỉ dùng lúc train, bỏ lúc inference.

| Model | Params | mAP50-95 |
|-------|--------|----------|
| v9-S | 7.2M | 46.8% |
| v9-E | 58.1M | **55.6%** |

**✅ Ưu:** PGI đột phá lý thuyết, ít params mà accuracy cao, lightweight models cải thiện rõ  
**❌ Nhược:** Training phức tạp, NMS, community nhỏ  
> *Tóm: Nhà khoa học — giải quyết vấn đề gốc rễ*

---

## 10. YOLOv10 (2024) — NMS-Free
**Tác giả:** Tsinghua University | **NMS-free** đầu tiên | Dual Assignment | CIB blocks

> **NMS-free:** Train nhiều box/object, inference 1 box/object → không cần lọc thêm.

| Model | Params | mAP50-95 |
|-------|--------|----------|
| v10-N | 2.3M | 38.5% |
| v10-X | 29.5M | 54.4% |

**✅ Ưu:** NMS-free (giảm latency), true end-to-end, ít params nhất (29.5M!)  
**❌ Nhược:** Training phức tạp, ecosystem non trẻ, ít multi-task  
> *Tóm: Dây chuyền tự động — bỏ kiểm tra thủ công*

---

## 11. YOLO11 (2024) — Nâng cấp toàn diện
**Tác giả:** Ultralytics | **C3K2** + **C2PSA** (Spatial Attention) | NMS-free

| Model | Params | mAP50-95 |
|-------|--------|----------|
| 11n | 2.6M | 39.5% |
| 11x | 56.9M | 54.7% |

**✅ Ưu:** Ít params hơn v8 mà accuracy cao hơn, multi-task + tracking, spatial attention  
**❌ Nhược:** Cải tiến incremental, không paper  
> *Tóm: Update iOS mới — mọi thứ tốt hơn 1 chút*

---

## 12. YOLOv12 (2025) — Kỷ nguyên Attention
**Tác giả:** Yunjie Tian et al. | **Area Attention (A²)** | **R-ELAN** | FlashAttention

Bản đầu tiên dùng **Attention** làm core thay CNN. Area Attention chia feature map thành 4 vùng → nhanh 4× so với global attention.

> **CNN** = đọc từng dòng sách | **Attention** = nhìn cả trang cùng lúc

| Model | Params | mAP50-95 |
|-------|--------|----------|
| v12-N | 2.6M | 40.6% |
| v12-X | 59.1M | 55.2% |

**✅ Ưu:** Attention-centric đầu tiên mà vẫn real-time, mAP cao nhất (tại thời điểm)  
**❌ Nhược:** Cần GPU hiện đại, edge deployment khó  
> *Tóm: Tesla — công nghệ mới nhất, cần hạ tầng phù hợp*

---

## 13. YOLOE (2025) — Open-Vocabulary
**Tác giả:** Tsinghua | Phát hiện vật **chưa từng train** | 3 chế độ

| Chế độ | Input | Ví dụ |
|--------|-------|-------|
| Text prompt | Gõ mô tả | "Tìm chiếc váy đỏ" |
| Visual prompt | Đưa ảnh mẫu | Đưa ảnh logo → tìm logo giống |
| Prompt-free | Không cần gì | Tự phát hiện 1200+ categories |

**Kỹ thuật:** RepRTA (text), SAVPE (visual), LRPC (prompt-free). Re-parameterize → không chậm thêm.

**✅ Ưu:** Detect vật mới không cần train lại, 3 modes, vượt YOLO-World  
**❌ Nhược:** Zero-shot accuracy thấp hơn closed-set, cần CLIP  
> *Tóm: Bảo vệ đa năng — mô tả, đưa ảnh, hoặc tự tìm đều được*

---

## 14. YOLO26 (2026) — Edge-First *(Mới nhất!)*
**Tác giả:** Ultralytics | NMS-free + Bỏ DFL | **MuSGD** | **STAL** | **ProgLoss**

- **MuSGD:** SGD + Muon (từ LLM training) → hội tụ nhanh hơn
- **STAL:** Phát hiện vật cực nhỏ (<8px) → drone, vệ tinh
- **ProgLoss:** Tự cân bằng loss theo tiến trình training
- **Nano nhanh hơn YOLO11 tới 43% trên CPU**

| Model | Params | mAP50-95 |
|-------|--------|----------|
| 26n | ~2.5M | ~40.5% |
| 26x | ~60M | **~57.2%** |

**✅ Ưu:** Nhanh nhất CPU/edge, mAP cao nhất YOLO family, export dễ nhất  
**❌ Nhược:** Mới (01/2026), bỏ DFL có thể giảm localization  
> *Tóm: Xe điện economy — tiết kiệm, nhanh, phù hợp đại chúng*

---

# PHẦN 2: BIẾN THỂ (Nhánh phụ)

---

## 15. YOLOX (2021) — Megvii
Anchor-free + Decoupled head + SimOTA tiên phong. YOLOX-L: 50.0% mAP, 69 FPS. 🥇 CVPR 2021.
> *Ảnh hưởng: Decoupled head → v6, v8+. Anchor-free → v6, v8+*

## 16. PP-YOLOE (2022) — Baidu
TAL label assignment + CSPRepResNet. PP-YOLOE+-L: 53.3% mAP, 78 FPS. Tối ưu PaddlePaddle.
> *Ảnh hưởng: TAL → YOLOv8, v9, v10+*

## 17. YOLO-NAS (2023) — Deci AI
AI tự thiết kế AI (AutoNAC). Quantization INT8 chỉ mất 0.4-0.6% mAP. YOLO-NAS-L: 52.2%.
> *Đặc biệt: Nén INT8 gần như không mất chất lượng*

## 18. YOLO-World (2024) — Tencent
Open-vocabulary đầu tiên cho YOLO. Text prompt detect. Tiền nhiệm YOLOE.
> *Ảnh hưởng: Mở đường open-vocab → YOLOE cải tiến*

## 19. Khác
- **DAMO-YOLO** (Alibaba, 2022): NAS + distillation, ~50% mAP
- **Gold-YOLO** (Huawei, 2023): Gather-and-Distribute fusion, 52.1% mAP
- **YOLOR** (2021): Implicit+explicit knowledge → tiền nhiệm YOLOv7
- **Scaled-YOLOv4** (2021): CSP scaling → tiền nhiệm YOLOv7

---

# PHẦN 3: SO SÁNH TỔNG HỢP

## Bảng master — Tất cả versions

| # | Version | Năm | mAP (best) | Anchor | NMS | Multi-Task | Dễ dùng |
|---|---------|-----|-----------|:------:|:---:|:----------:|:-------:|
| 1 | v1 | 2015 | 33% | ❌ | Cần | ❌ | ⭐⭐ |
| 2 | v2 | 2016 | 34% | ✅ | Cần | ❌ | ⭐⭐ |
| 3 | v3 | 2018 | 33% | ✅ | Cần | ❌ | ⭐⭐ |
| 4 | v4 | 2020 | 43.5% | ✅ | Cần | ❌ | ⭐⭐ |
| 5 | v5 | 2020 | 50.7% | ✅ | Cần | ⚠️ | ⭐⭐⭐⭐⭐ |
| 6 | v6 | 2022 | 52.5% | ❌ | Cần | ❌ | ⭐⭐⭐ |
| 7 | v7 | 2022 | 56.8% | ✅ | Cần | ⚠️ | ⭐⭐⭐ |
| 8 | v8 | 2023 | 53.9% | ❌ | Cần | ✅ | ⭐⭐⭐⭐⭐ |
| 9 | v9 | 2024 | 55.6% | ❌ | Cần | ⚠️ | ⭐⭐⭐ |
| 10 | v10 | 2024 | 54.4% | ❌ | ❌ | ❌ | ⭐⭐⭐ |
| 11 | v11 | 2024 | 54.7% | ❌ | ❌ | ✅ | ⭐⭐⭐⭐⭐ |
| 12 | v12 | 2025 | 55.2% | ❌ | ❌ | ✅ | ⭐⭐⭐⭐ |
| 13 | YOLOE | 2025 | 27%* | ❌ | Tùy | ✅ | ⭐⭐⭐ |
| 14 | YOLO26 | 2026 | **57.2%** | ❌ | ❌ | ✅ | ⭐⭐⭐⭐⭐ |
| — | YOLOX | 2021 | 51.2% | ❌ | Cần | ❌ | ⭐⭐⭐ |
| — | PP-YOLOE | 2022 | 53.3% | ❌ | Cần | ❌ | ⭐⭐ |
| — | YOLO-NAS | 2023 | 52.2% | ❌ | Cần | ❌ | ⭐⭐⭐ |
| — | YOLO-World | 2024 | 32.5%* | ❌ | Cần | ⚠️ | ⭐⭐⭐ |

*\* LVIS zero-shot — thang đo khác COCO closed-set*

## Xu hướng phát triển

```
Backbone:  CNN → Darknet → CSPDarknet → ELAN → GELAN → Attention
Anchor:    Không → Có (v2-v7) → Anchor-free (v6,v8+) → NMS-free (v10+)
Loss:      MSE → IoU → CIoU → DFL → Bỏ DFL (YOLO26)
Framework: Darknet(C) → PyTorch → Ultralytics pip install
Task:      Detection only → +Seg+Cls → +Pose+OBB+Track
Vocab:     Closed-set → Open-vocabulary (YOLO-World, YOLOE)
```

## Sơ đồ ảnh hưởng

```
YOLOv1→v2→v3 (Joseph Redmon)
      ↓
YOLOv4 (Bochkovskiy) ←── YOLOR, Scaled-v4 ──→ YOLOv7 (Wang)
      ↓                                            ↓
YOLOv5 (Ultralytics) ──→ YOLOv8 ──→ YOLO11 ──→ YOLO26
      ↑                    ↑
YOLOX (decoupled head)    PP-YOLOE (TAL)
                           ↑
                    YOLO-World ──→ YOLOE (open-vocab)
```

---

# PHẦN 4: HƯỚNG DẪN CHỌN PHIÊN BẢN

| Mục đích | Phiên bản | Lý do |
|----------|----------|-------|
| **Sinh viên mới học** | YOLOv3 | Paper rõ ràng, dễ hiểu |
| **Dự án nhanh, dễ dùng** | YOLOv8 / YOLO11 | Ecosystem tốt nhất, 3 dòng code |
| **Accuracy cao nhất** | YOLO26-X / YOLOv9-E | mAP 57% / 55.6% |
| **Deploy điện thoại/edge** | YOLO26-N / YOLOv10-N | NMS-free, siêu nhẹ, CPU nhanh |
| **Real-time cực nhanh** | YOLOv6-S | 520 FPS |
| **Detect + Segment + Pose** | YOLO11 / YOLOv8 | Multi-task native |
| **Nhà máy/công nghiệp** | YOLOv6 / YOLO-NAS | INT8/TensorRT tối ưu |
| **Detect vật chưa train** | YOLOE | Open-vocab, 3 prompt modes |
| **Vật nhỏ (drone, vệ tinh)** | YOLO26 | STAL chuyên vật nhỏ |
| **Quantization INT8** | YOLO-NAS | Mất chỉ 0.4% mAP |
| **Nghiên cứu baseline** | YOLOX | Paper chuẩn, anchor-free |
| **PaddlePaddle ecosystem** | PP-YOLOE | Tối ưu Baidu framework |

---

*Tài liệu tổng hợp từ tất cả papers gốc. Cập nhật 05/2026.*
*Sẵn sàng thêm phần khuyến nghị cho bài toán cụ thể của bạn.*
