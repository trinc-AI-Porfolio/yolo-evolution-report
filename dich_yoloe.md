# DỊCH CHI TIẾT PAPER YOLOE
**"YOLOE: Real-Time Seeing Anything"**
**Hội nghị:** ICCV 2025 | **Link:** https://arxiv.org/abs/2503.07465

---

## ABSTRACT

Các open-vocabulary detectors hiện tại (GLIP, Grounding DINO) mạnh nhưng quá chậm cho real-time. Các YOLO models nhanh nhưng chỉ detect được classes đã train (closed-set). YOLOE kết hợp cả hai: **open-vocabulary + real-time** thông qua 3 modules: RepRTA (text prompt), SAVPE (visual prompt), LRPC (prompt-free). YOLOE đạt 27.1% AP trên LVIS zero-shot (vượt YOLO-World-v2 22.7%) với inference speed tương đương YOLO closed-set.

---

## 1. INTRODUCTION

**Closed-set detection:** Nhanh nhưng chỉ detect classes đã train. Thêm class mới = train lại từ đầu.

**Open-vocabulary detection:** Detect bất kỳ vật nào qua text/visual prompt. Nhưng tất cả models hiện tại (Grounding DINO, GLIP) đều quá chậm cho real-time vì dùng language models nặng.

**Mục tiêu YOLOE:** Open-vocabulary mà vẫn real-time, 3 chế độ:
1. **Text prompt:** Mô tả bằng text → detect
2. **Visual prompt:** Đưa ảnh mẫu → tìm vật tương tự
3. **Prompt-free:** Tự detect mọi thứ, không cần input

---

## 2. METHOD

### 2.1. RepRTA (Re-parameterizable Region-Text Alignment)

**Cho chế độ Text Prompt.**

Cách hoạt động:
1. Text input (ví dụ "red car") → qua **CLIP text encoder** → text embeddings
2. Lightweight projection network tinh chỉnh embeddings cho phù hợp detection task
3. Region features từ detector so khớp với text embeddings → tìm vật phù hợp

**Re-parameterization trick:**
- Training: Có projection network riêng (auxiliary branch)
- Inference: **Gộp projection vào model chính** → zero overhead
- Nghĩa là: text embeddings được "bake" vào weights → inference nhanh bằng closed-set YOLO

### 2.2. SAVPE (Semantic-Activated Visual Prompt Encoder)

**Cho chế độ Visual Prompt.**

Khi user đưa 1 ảnh mẫu (ví dụ: ảnh logo 1xBet), SAVPE trích xuất 2 loại features:

**1. Semantic features:** "Đây là CÁI GÌ?" — ý nghĩa tổng quát (logo, có chữ, màu xanh...)
**2. Activation features:** "Nó TRÔNG thế nào?" — chi tiết cụ thể (font chữ, viền, pattern...)

Hai features này kết hợp qua **cross-attention** với detection features → tìm vùng khớp trong ảnh đầu vào.

Tại sao cần cả 2:
- Chỉ semantic → biết tìm "logo" nhưng không biết logo NÀO
- Chỉ activation → biết pattern nhưng nếu bị biến dạng → không nhận ra
- Cả 2 → vừa hiểu loại vật, vừa biết chi tiết → chính xác hơn

### 2.3. LRPC (Lazy Region-Prompt Contrast)

**Cho chế độ Prompt-Free.**

Detect hàng nghìn categories mà không cần user input:
1. Pre-compute text embeddings cho vocabulary lớn (LVIS 1203 classes + Objects365)
2. Cache embeddings → không cần chạy text encoder mỗi frame
3. Region features so khớp với cached embeddings bằng **similarity lookup** → nhanh

"Lazy" = tính embeddings 1 lần, dùng mãi → giảm computation đáng kể.

### 2.4. Training Strategy

**Pre-training:** Objects365 + GoldG (grounding data)
**Fine-tuning:** LVIS dataset (1203 categories)
**Supports:** Cả detection và instance segmentation

---

## 3. EXPERIMENTS

### Zero-shot trên LVIS:

| Model | AP (LVIS) | Inference |
|-------|----------|-----------|
| Grounding DINO-L | 27.4% | Rất chậm |
| YOLO-World-v2-L | 22.7% | Trung bình |
| **YOLOE-L** | **27.1%** | **Nhanh (real-time)** |

YOLOE gần bằng Grounding DINO về accuracy nhưng **nhanh hơn rất nhiều**.

### Transfer to COCO:

| Model | mAP50-95 |
|-------|---------|
| YOLO-World-L (finetune) | 45.7% |
| **YOLOE-L (finetune)** | **52.6%** |

Khi finetune trên COCO closed-set, YOLOE vượt YOLO-World đáng kể.

### Visual Prompt Results:

SAVPE cho phép detect vật thể từ ảnh mẫu mà không cần train. Hữu ích cho: tìm sản phẩm tương tự, detect logo cụ thể, surveillance.

---

## 4. CONCLUSION

YOLOE là model open-vocabulary real-time đầu tiên đạt accuracy cạnh tranh với heavy models (Grounding DINO) trong khi giữ tốc độ YOLO. 3 modules (RepRTA, SAVPE, LRPC) cho 3 chế độ sử dụng khác nhau. Re-parameterization là chìa khóa: gộp auxiliary components vào model → zero inference overhead.

---

*Bản dịch chi tiết từ paper gốc arXiv:2503.07465.*
