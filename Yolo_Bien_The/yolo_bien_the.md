# Các Biến Thể YOLO (YOLO Variants) — Nhánh Phụ

Ngoài dòng chính (v1→v12, YOLOE, YOLO26), nhiều công ty lớn đã tạo ra phiên bản YOLO riêng. Đây là tài liệu chi tiết về từng biến thể.

---

## 1. YOLOX (2021) — "Anchor-Free tiên phong"

**Tác giả:** Megvii (旷视科技) — công ty AI đứng sau Face++  
**Năm:** 07/2021  |  **Paper:** "YOLOX: Exceeding YOLO Series in 2021"  
**Giải thưởng:** 🥇 1st Place — CVPR 2021 Streaming Perception Challenge

### YOLOX đóng góp gì?

YOLOX là model **đầu tiên** chứng minh rằng YOLO có thể hoạt động tốt **không cần anchor boxes**. Trước đó, ai cũng nghĩ anchor là bắt buộc.

### Các kỹ thuật chính

#### a) Anchor-Free
Bỏ hoàn toàn anchor boxes. Model dự đoán trực tiếp offset từ tâm grid cell.

> **Ví dụ:** Trước đây (anchor-based) giống như bạn có sẵn 9 khung ảnh (ngang, dọc, vuông, lớn, nhỏ...) rồi chọn khung phù hợp nhất để đặt ảnh vào. YOLOX bỏ khung ảnh luôn — cứ vẽ khung tự do theo kích thước thật.

#### b) Decoupled Head — Tách riêng 2 nhiệm vụ
Lần đầu tiên trong YOLO: tách classification và regression thành 2 nhánh riêng.

> **Ví dụ:** Trong bệnh viện, trước đây 1 bác sĩ vừa chẩn đoán bệnh vừa chỉ vị trí đau. YOLOX chia thành 2 bác sĩ chuyên môn: 1 người chẩn đoán (classification), 1 người chỉ vị trí (regression) → mỗi người làm tốt hơn.

*Kỹ thuật này sau đó được YOLOv6, v8, v11, v12 đều adopt.*

#### c) SimOTA — Gán nhãn thông minh
SimOTA tự quyết định mỗi prediction nên được gán cho object nào, dựa trên chất lượng prediction hiện tại.

> **Ví dụ:** Giống phân công việc trong nhóm. Thay vì gán cố định (bạn A luôn làm việc 1), SimOTA đánh giá ai đang làm tốt nhất việc gì → gán linh hoạt. Kết quả: nhóm hoạt động hiệu quả hơn.

### Hiệu suất

| Model | Params | mAP (COCO) | FPS (V100) |
|-------|--------|-----------|------------|
| YOLOX-Nano | 0.91M | 25.3% | Cực nhanh |
| YOLOX-Tiny | 5.06M | 32.8% | Rất nhanh |
| YOLOX-S | 9.0M | 40.5% | Nhanh |
| YOLOX-M | 25.3M | 46.9% | Vừa |
| YOLOX-L | 54.2M | 50.0% | 68.9 FPS |
| YOLOX-X | 99.1M | 51.2% | Chậm hơn |

### ✅ Ưu điểm
1. ✅ **Chứng minh anchor-free khả thi** → mở đường cho v6, v8+
2. ✅ **Decoupled head** → kỹ thuật chuẩn từ đây về sau
3. ✅ **SimOTA** — label assignment thông minh, động
4. ✅ **Nano variant cực nhỏ** (0.91M params!) — IoT, embedded
5. ✅ **Paper chất lượng** — baseline phổ biến trong nghiên cứu

### ❌ Nhược điểm
1. ❌ **Ecosystem hạn chế** — không có unified framework như Ultralytics
2. ❌ **Đã bị vượt** bởi YOLOv8+ về accuracy
3. ❌ **Ít multi-task** — chỉ detection

> **Tóm lại:** YOLOX như nhà phát minh — tạo ra anchor-free và decoupled head mà tất cả YOLO sau đều sử dụng. "Người mở đường" thầm lặng.

---

## 2. PP-YOLOE / PP-YOLOE+ (2022) — "YOLO cho nhà máy Trung Quốc"

**Tác giả:** Baidu (百度) — công ty search engine lớn nhất Trung Quốc  
**Năm:** 2022  |  **Framework:** PaddlePaddle (Baidu's deep learning framework)

### PP-YOLOE là gì?

PP = PaddlePaddle. Đây là YOLO được Baidu tối ưu để chạy trên hệ sinh thái PaddlePaddle — framework AI của Baidu dùng rộng rãi trong công nghiệp Trung Quốc.

> **Ví dụ:** Như Toyota tạo ra phiên bản xe riêng tối ưu cho đường Nhật Bản. PP-YOLOE là YOLO tối ưu cho "con đường" PaddlePaddle.

### Kỹ thuật chính

#### a) Anchor-Free + ET-Head
- Anchor-free giống YOLOX
- **ET-Head (Efficient Task-aligned Head):** Head hiệu quả hơn decoupled head

#### b) TAL (Task Alignment Learning)
Chiến lược gán nhãn căn chỉnh giữa classification score và localization quality.

> **Ví dụ:** Trước đây, model có thể rất tự tin "đây là mèo" (classification cao) nhưng vẽ box sai vị trí (localization kém). TAL đảm bảo: tự tin cao ↔ box chính xác phải đi đôi.

#### c) CSPRepResNet Backbone
Kết hợp CSP + Re-parameterization + Residual connections.

### Hiệu suất

| Model | mAP (COCO) | FPS |
|-------|-----------|-----|
| PP-YOLOE-S | 43.1% | 208 FPS (V100 TRT) |
| PP-YOLOE-M | 48.9% | 123 FPS |
| PP-YOLOE-L | 51.4% | 78 FPS |
| PP-YOLOE-X | 52.2% | 45 FPS |
| PP-YOLOE+-L | 53.3% | 78 FPS |

### ✅ Ưu điểm
1. ✅ **Cực nhanh** — 208 FPS cho Small variant!
2. ✅ **TAL** — kỹ thuật được YOLOv8+ adopt
3. ✅ **Production-ready** — nhiều nhà máy Trung Quốc sử dụng
4. ✅ **Paper chất lượng**, có peer review

### ❌ Nhược điểm
1. ❌ **Phụ thuộc PaddlePaddle** — phần lớn devs dùng PyTorch
2. ❌ **Community quốc tế nhỏ** — phổ biến chủ yếu ở Trung Quốc
3. ❌ **Khó tích hợp** nếu bạn không dùng PaddlePaddle

> **Tóm lại:** PP-YOLOE giống WeChat — cực mạnh ở Trung Quốc nhưng ít phổ biến ở nước ngoài.

---

## 3. YOLO-NAS (2023) — "AI tự thiết kế AI"

**Tác giả:** Deci AI (Israel)  
**Năm:** 05/2023  |  **Thư viện:** SuperGradients (open-source)

### YOLO-NAS đặc biệt ở chỗ nào?

NAS = **Neural Architecture Search** — thay vì con người tự thiết kế kiến trúc model, dùng **AI để tìm kiến trúc tối ưu**.

> **Ví dụ:** Các YOLO khác giống như kiến trúc sư tự vẽ bản thiết kế nhà. YOLO-NAS giống như thuê 1 siêu máy tính thử hàng ngàn tỷ bản thiết kế khác nhau rồi chọn ra cái tốt nhất. AI thiết kế AI!

### Kỹ thuật chính

#### a) AutoNAC — Tự tìm kiến trúc tối ưu
- Khám phá ~10^14 (10 triệu tỷ) kiến trúc khác nhau
- Tự quyết định: bao nhiêu layers, bao nhiêu channels, block nào ở đâu
- Tối ưu cho hardware cụ thể (GPU, CPU, edge...)

> **Ví dụ:** Như dùng GPS thử 10 triệu con đường rồi tìm ra đường ngắn nhất cho bạn, thay vì bạn tự đoán đường đi.

#### b) Quantization-Aware Blocks (QSP, QCI)
Thiết kế sẵn để chạy tốt khi **quantize xuống INT8** (giảm từ 32-bit xuống 8-bit).

> **Ví dụ:** Hầu hết model khi "nén" xuống 8-bit đều mất accuracy (giống nén ảnh JPEG bị mờ). YOLO-NAS được thiết kế để khi nén vẫn giữ chất lượng — giống nén ảnh mà không bị mờ.

**Kết quả quantization:**
| Model | mAP (FP32) | mAP (INT8) | Mất bao nhiêu? |
|-------|-----------|-----------|-----------------|
| YOLO-NAS-S | 47.5% | 47.0% | Chỉ mất **0.5%** |
| YOLO-NAS-M | 51.5% | 50.9% | Chỉ mất **0.6%** |
| YOLO-NAS-L | 52.2% | 51.8% | Chỉ mất **0.4%** |

> Các model khác thường mất 1-2% khi quantize!

### ✅ Ưu điểm
1. ✅ **AI tự thiết kế** — kiến trúc tối ưu hơn thiết kế thủ công
2. ✅ **Quantization xuất sắc** — INT8 gần như không mất accuracy
3. ✅ **Tốt cho production/edge** — deploy trên hardware thật hiệu quả
4. ✅ **Pre-trained trên data lớn** (COCO + Objects365 + Roboflow 100)

### ❌ Nhược điểm
1. ❌ **Closed-source architecture** — không biết chính xác kiến trúc bên trong
2. ❌ **SuperGradients framework** — không phổ biến bằng Ultralytics
3. ❌ **Đã bị vượt** accuracy bởi YOLOv9+
4. ❌ **Ít multi-task** — chủ yếu detection

> **Tóm lại:** YOLO-NAS như nhà do AI thiết kế — tối ưu hoàn hảo cho mục đích cụ thể, nhưng bạn không biết tại sao nó lại được thiết kế như vậy.

---

## 4. YOLO-World (2024) — "Tiền nhiệm của YOLOE"

**Tác giả:** Tencent AI Lab (腾讯)  
**Năm:** 02/2024  |  **Paper:** "YOLO-World: Real-Time Open-Vocabulary Object Detection"

### YOLO-World là gì?

Giống YOLOE, YOLO-World cũng là **open-vocabulary** — phát hiện vật chưa train. Nhưng YOLO-World ra trước YOLOE 1 năm.

> **Ví dụ:** YOLO-World là iPhone 15, YOLOE là iPhone 16 — cùng ý tưởng nhưng đời sau tốt hơn.

### Kỹ thuật chính

#### a) Re-parameterizable Vision-Language PAN (RepVL-PAN)
Kết hợp **text embeddings** (từ CLIP) vào **feature pyramid** để model hiểu cả hình ảnh lẫn ngôn ngữ.

> **Ví dụ:** Như dạy con robot vừa nhìn vừa đọc. Robot không chỉ nhận diện hình dạng mà còn hiểu mô tả bằng lời: "tìm cái bàn gỗ tròn có chân sắt."

#### b) Region-Text Contrastive Learning
Train model bằng cách so khớp vùng ảnh với mô tả text — vùng nào khớp text nào.

#### c) Prompt-then-Detect
- Bước 1: Nhập text prompt (offline) → tạo embeddings
- Bước 2: Detect real-time dùng embeddings đã tạo

> **Ví dụ:** Bạn nạp sẵn "helmet, vest, boot" vào hệ thống 1 lần. Sau đó camera nhà máy tự detect 3 thứ đó real-time mà không cần train lại.

### Hiệu suất

| Model | Zero-shot mAP (LVIS) | FPS |
|-------|----------------------|-----|
| YOLO-World-S | 22.7% | 74 FPS |
| YOLO-World-M | 28.7% | 52 FPS |
| YOLO-World-L | 32.5% | 35 FPS |

### So sánh YOLO-World vs YOLOE

| Tiêu chí | YOLO-World | YOLOE |
|----------|-----------|-------|
| Năm | 02/2024 | 03/2025 |
| Zero-shot (LVIS-S) | 22.7% | **27.1%** (+4.4%) |
| Visual prompt | ❌ | ✅ |
| Prompt-free | ❌ | ✅ |
| Re-param overhead | Có chút | **Không** (zero overhead) |
| Segmentation | ❌ | ✅ |

### ✅ Ưu điểm
1. ✅ **Mở đường open-vocabulary cho YOLO** — ý tưởng đột phá
2. ✅ **Prompt-then-Detect** — customize detection không cần train
3. ✅ **Tích hợp Ultralytics** — dùng được trong ecosystem v8

### ❌ Nhược điểm
1. ❌ **Chỉ có text prompt** — không có visual prompt hay prompt-free
2. ❌ **Đã bị YOLOE vượt** về accuracy và tính năng
3. ❌ **Inference overhead** — chậm hơn YOLO thường

> **Tóm lại:** YOLO-World là Columbus — phát hiện "lục địa mới" (open-vocabulary) cho YOLO. YOLOE là người đến sau xây dựng trên nền tảng đó.

---

## 5. Các biến thể khác (tóm tắt)

### DAMO-YOLO (2022) — Alibaba
- Dùng NAS (Neural Architecture Search) + distillation
- Tối ưu cho cloud Alibaba
- **Đặc biệt:** RepGFPN neck, AlignedOTA label assignment
- mAP: ~50% (COCO), cạnh tranh YOLOv7

### Gold-YOLO (2023) — Huawei
- **Gather-and-Distribute (GD)** mechanism cho feature fusion
- Thay thế FPN/PAN truyền thống → feature fusion hiệu quả hơn
- mAP: ~52.1% (COCO-L variant)

### YOLOR (2021) — Chien-Yao Wang
- **Implicit + Explicit knowledge** kết hợp
- Model học cả kiến thức hiển (features) lẫn kiến thức ẩn (priors)
- Tiền nhiệm của YOLOv7

### Scaled-YOLOv4 (2021) — Chien-Yao Wang
- Scale YOLOv4 cho cả **cloud** (lớn) và **edge** (nhỏ)
- CSP connections + model scaling strategy
- Tiền nhiệm của YOLOv7

### PP-YOLO / PP-YOLOv2 (2020-2021) — Baidu
- YOLOv3 cải tiến cho PaddlePaddle
- Tổng hợp nhiều tricks (Deformable Conv, Matrix NMS, CoordConv...)
- Tiền nhiệm của PP-YOLOE

---

## 6. Bảng so sánh tổng hợp tất cả biến thể

| Model | Tác giả | Năm | mAP (best) | Đặc trưng | Open-Vocab |
|-------|---------|-----|-----------|-----------|:----------:|
| YOLOX | Megvii | 2021 | 51.2% | Anchor-free tiên phong | ❌ |
| PP-YOLOE+ | Baidu | 2022 | 53.3% | Industrial, TAL | ❌ |
| DAMO-YOLO | Alibaba | 2022 | ~50% | NAS + Distillation | ❌ |
| YOLO-NAS | Deci | 2023 | 52.2% | AI thiết kế, Quantization | ❌ |
| Gold-YOLO | Huawei | 2023 | 52.1% | GD mechanism | ❌ |
| YOLO-World | Tencent | 2024 | 32.5%* | Open-vocab, text prompt | ✅ |
| **YOLOE** | Tsinghua | 2025 | 27.1%* | 3 prompt modes | ✅ |

*\* LVIS zero-shot (khác scale so với COCO closed-set)*

### Ai ảnh hưởng ai?

```
YOLOX (2021) ──→ Anchor-free ──→ YOLOv6, v8, v10+
        └──→ Decoupled head ──→ YOLOv6, v8, v11, v12

PP-YOLOE (2022) ──→ TAL ──→ YOLOv8, v9, v10+

YOLOR (2021) ──→ Implicit knowledge ──→ YOLOv7
Scaled-YOLOv4 ──→ CSP scaling ──→ YOLOv7

YOLO-NAS (2023) ──→ Quantization-aware ──→ YOLO26 (INT8 focus)

YOLO-World (2024) ──→ Open-vocab ──→ YOLOE (cải tiến)
```

### Khi nào chọn biến thể nào?

| Tình huống | Chọn | Lý do |
|------------|------|-------|
| Nghiên cứu anchor-free | **YOLOX** | Paper chuẩn, baseline phổ biến |
| Nhà máy dùng PaddlePaddle | **PP-YOLOE** | Tối ưu cho PaddlePaddle |
| Deploy INT8 trên edge | **YOLO-NAS** | Quantization tốt nhất |
| Open-vocab trong Ultralytics | **YOLO-World** | Tích hợp sẵn v8 ecosystem |
| Open-vocab tốt nhất | **YOLOE** | 3 prompt modes, vượt YOLO-World |
| Feature fusion nghiên cứu | **Gold-YOLO** | GD mechanism mới lạ |
| Cloud deployment Alibaba | **DAMO-YOLO** | Tối ưu Alibaba Cloud |

---

*Tài liệu cập nhật 05/2026.*
