# Chương 5: THẾ HỆ MỚI — YOLOv9–v12 (2024–2025)

## 5.1. YOLOv9 (2024) — Chống mất thông tin

**Paper:** "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information" [9]  
**Tác giả:** Chien-Yao Wang, I-Hau Yeh, Hong-Yuan Mark Liao

### 5.1.1. Vấn đề: Information Bottleneck

Khi data đi qua nhiều layers, **mutual information giảm đơn điệu**:

$$I(X, X) \geq I(X, f_1(X)) \geq I(X, f_2(f_1(X))) \geq \ldots$$

> **Ẩn dụ:** Giống trò **truyền tin** — người đầu nhận tin gốc, truyền miệng qua 20 người, người cuối nhận tin sai lệch. Layers sâu "quên" thông tin input gốc → gradient không đáng tin cậy [9].

### 5.1.2. PGI (Programmable Gradient Information)

3 thành phần:

1. **Main Branch:** Network chính, dùng cho inference. Kiến trúc tùy chọn.
2. **Auxiliary Reversible Branch:** Mạng phụ có khả năng **khôi phục input từ output** → giữ complete information. **Chỉ dùng khi training, bỏ khi inference** → zero cost.
3. **Multi-level Auxiliary:** Inject gradient ở **nhiều tầng** — tránh gradient bị "đứt".

> **Ẩn dụ:** Main branch giống "truyền miệng" qua 20 người (mất thông tin). PGI thêm 1 **kênh SMS song song** (auxiliary branch) — mỗi người đều nhận SMS đối chiếu → sửa sai kịp thời. SMS **chỉ dùng lúc luyện tập**, khi "lên sân khấu" (inference) thì bỏ.

[📌 YÊU CẦU CHÈN ẢNH: Tác giả hãy chèn sơ đồ PGI (Main Branch + Auxiliary Reversible Branch) từ paper YOLOv9 (Figure 4) [9].]

### 5.1.3. GELAN (Generalized ELAN)

Tổng quát hóa CSPNet + ELAN — cho phép dùng **bất kỳ computational block** nào (Conv, Bottleneck, Transformer...) [9].

> **Ẩn dụ:** ELAN cũ giống dây chuyền chỉ lắp được **1 loại máy**. GELAN giống dây chuyền "universal" — lắp máy gì vào cũng chạy được.

### 5.1.4. Kết quả

| Model | Params | FLOPs | mAP@50:95 | So với v8 |
|:--|:-:|:-:|:-:|:--|
| v8-S | 11.2M | 28.6G | 44.9% | — |
| **v9-S** | **7.2M** | **26.7G** | **46.8%** | +1.9%, ít params hơn |
| v8-M | 25.9M | 78.9G | 50.2% | — |
| **v9-C** | **25.5M** | **102.8G** | **53.0%** | +2.8% |
| v8-X | 68.2M | 257.8G | 53.9% | — |
| **v9-E** | **58.1M** | **192.5G** | **55.6%** | +1.7%, ít params hơn |

*Ghi chú: Đo trên COCO val2017. v9 luôn ít params hơn v8 mà accuracy cao hơn ở mọi scale — chứng minh PGI hiệu quả [9].*

---

## 5.2. YOLOv10 (2024) — NMS-Free

**Paper:** "YOLOv10: Real-Time End-to-End Object Detection" [10]  
**Tác giả:** Ao Wang et al. (Tsinghua University)

### 5.2.1. Vấn đề NMS

NMS có 3 nhược điểm: (1) Cần tune threshold, (2) Không differentiable, (3) Latency phụ thuộc data — ảnh nhiều vật → NMS chậm hơn → **latency không ổn định** [10].

> **Ẩn dụ:** NMS giống **kiểm tra chất lượng cuối dây chuyền** — phải dừng lại soi từng sản phẩm, loại phế phẩm. v10 thiết kế dây chuyền "zero defect" — không cần kiểm tra cuối.

### 5.2.2. Consistent Dual Assignments

- **One-to-many (o2m):** 1 vật → nhiều predictions → nhiều training signal → train tốt
- **One-to-one (o2o):** 1 vật → 1 prediction → không cần NMS → deploy tốt
- **Training:** Dùng **cả 2 heads**. **Inference:** Chỉ dùng **o2o head** → NMS-free [10]

### 5.2.3. Kết quả

| Model | Params | Latency | mAP@50:95 | Ghi chú |
|:--|:-:|:-:|:-:|:--|
| v8-S + NMS | 11.2M | 4.70ms | 44.9% | Cần NMS |
| **v10-S (Dual)** | **7.2M** | **2.49ms** | **46.3%** | NMS-free, nhanh 2× |
| v8-X | 68.2M | 7.61ms | 53.9% | — |
| **v10-X** | **29.5M** | **10.70ms** | **54.4%** | 2.3× ít params |

⚠️ **Trade-off:** v10-X (54.4%) **thua** v9-E (55.6%) về mAP. Nhưng v10 ít params gần **2×** và **không cần NMS** — đổi accuracy lấy efficiency [9][10].

---

## 5.3. YOLO11 (2024) — Nâng cấp toàn diện

**Tác giả:** Ultralytics | *Không có paper* [11]  
*Lưu ý:* Bỏ "v" trong tên — **YOLO11** thay vì YOLOv11.

### 5.3.1. C3K2 (CSP Bottleneck with 2 Kernel sizes)

Dùng **2 kernel sizes: $3 \times 3$ và $5 \times 5$** đồng thời:
- Kernel $3 \times 3$: Capture **local features** (chi tiết nhỏ, cạnh, góc)
- Kernel $5 \times 5$: Capture **wider context** (vùng rộng hơn)

> **Ẩn dụ:** C2f giống nhìn qua **1 ống kính cố định**. C3K2 giống nhìn qua **2 ống kính zoom khác nhau cùng lúc** — vừa thấy chi tiết, vừa thấy bối cảnh.

### 5.3.2. C2PSA (Cross Stage Partial Spatial Attention)

Thêm **spatial attention** vào CSP — học "vùng nào quan trọng", highlight vùng có vật thể, suppress background.

> **Ẩn dụ:** C2f giống quét đều toàn bộ ảnh (mỗi pixel xử lý ngang nhau). C2PSA giống **mắt người** — tập trung vào vùng quan trọng, lướt nhanh qua background.

### 5.3.3. Kết quả

| Model | Params | mAP@50:95 | So với v8 |
|:--|:-:|:-:|:--|
| YOLO11n | 2.6M | 39.5% | v8n: 37.3% (**+2.2%**) |
| YOLO11s | 9.4M | 47.0% | v8s: 44.9% (**+2.1%**) |
| YOLO11m | 20.1M | 51.5% | v8m: 50.2% (**+1.3%**) |
| YOLO11l | 25.3M | 53.4% | v8l: 52.9% (**+0.5%**) |
| YOLO11x | 56.9M | 54.7% | v8x: 53.9% (**+0.8%**) |

---

## 5.4. YOLOv12 (2025) — Kỷ nguyên Attention

**Paper:** "YOLOv12: Attention-Centric Real-Time Object Detectors" [12]  
**Tác giả:** Yunjie Tian, Qixiang Ye, David Doermann

### 5.4.1. Thách thức

Self-attention có complexity $O((H \times W)^2)$ — quá chậm cho real-time. v12 giải quyết bằng **Area Attention** [12].

> **Ẩn dụ:** CNN giống đọc sách **từng dòng** (local, nhanh). Attention giống nhìn **cả trang cùng lúc** (global, hiểu sâu nhưng chậm). v12 tìm cách "nhìn cả trang" mà vẫn nhanh.

### 5.4.2. Area Attention (A²)

Chia feature map thành **n vùng bằng nhau**, tính attention trong từng vùng:

$$O(\text{Global}) = (HW)^2 \quad \rightarrow \quad O(\text{Area}, n=4) = HW \times \frac{HW}{4}$$

Giảm **4× complexity**, chỉ mất **0.1% AP** [12].

> **Ẩn dụ:** Global attention giống 1 giáo viên kiểm tra bài **tất cả** 100 học sinh cùng lúc (quá tải). Area attention chia 4 nhóm 25 người, kiểm tra từng nhóm → nhanh 4×, kết quả gần tương đương.

### 5.4.3. R-ELAN (Residual ELAN)

**Vấn đề:** Thay CNN bằng attention trong ELAN → training bất ổn. **Giải pháp:** Thêm residual shortcut với **scale factor 0.01** → gradient ổn định ở đầu training [12].

### 5.4.4. Kết quả

| Model | Params | GFLOPs | mAP@50:95 |
|:--|:-:|:-:|:-:|
| YOLO11-N | 2.6M | 6.5 | 39.5% |
| **YOLOv12-N** | **2.6M** | **6.5** | **40.6%** |
| YOLO11-S | 9.4M | 21.5 | 47.0% |
| **YOLOv12-S** | **9.3M** | **21.4** | **48.0%** |
| **YOLOv12-X** | **59.1M** | **199.0** | **55.2%** |

**Ý nghĩa:** v12 đánh dấu YOLO chuyển từ **CNN-centric** sang **Attention-centric** [12]. Nhược điểm: cần GPU hỗ trợ FlashAttention.

---

# Chương 6: MỞ RỘNG — YOLOE & YOLO26

## 6.1. YOLOE (2025) — Open-Vocabulary Real-Time

**Paper:** "YOLOE: Real-Time Seeing Anything" (ICCV 2025) [13]

### 6.1.1. Vấn đề

Closed-set YOLO chỉ detect classes đã train. Open-vocabulary detectors (Grounding DINO) detect mọi thứ nhưng quá chậm. YOLOE kết hợp: **open-vocabulary + real-time** [13].

> **Ẩn dụ:** YOLO truyền thống giống **nhân viên bảo vệ chỉ biết nhận 80 khuôn mặt đã học**. YOLOE giống bảo vệ được đào tạo **nhận diện bất kỳ ai** — chỉ cần bạn mô tả hoặc đưa ảnh mẫu.

### 6.1.2. Ba chế độ Prompt

| Chế độ | Input | Module | Cơ chế |
|:--|:--|:--|:--|
| **Text Prompt** | "dog", "car" | RepRTA | CLIP text embeddings → so khớp region features |
| **Visual Prompt** | Ảnh mẫu | SAVPE | Trích semantic + activation features → cross-attention |
| **Prompt-Free** | Không cần | LRPC | Pre-compute 1203 categories, cache → similarity lookup |

**Kết quả:** 27.1% AP trên LVIS zero-shot — vượt YOLO-World-v2 (22.7%), gần Grounding DINO (27.4%) nhưng **nhanh hơn rất nhiều** [13].

---

## 6.2. YOLO26 (2026) — Edge-First Design

**Tác giả:** Ultralytics | *Không có paper chính thức* [14]  
*Phiên bản mới nhất tính đến thời điểm viết báo cáo.*

### 6.2.0. Tại sao gọi là "YOLO26" thay vì "YOLOv13"?

Từ YOLO11 (2024), Ultralytics đã bỏ tiền tố "v" (version). Đến phiên bản tiếp theo, thay vì đặt "YOLO12" (trùng với YOLOv12 của nhóm khác), Ultralytics quyết định dùng **quy tắc đặt tên theo năm phát hành**: YOLO**26** = phiên bản phát hành năm 20**26**. Cách đặt tên này tránh xung đột số phiên bản và cho thấy ngay thời điểm release [14].

### 6.2.1. Bỏ DFL — Thay đổi triết lý lớn nhất

DFL (Distribution Focal Loss, v8–v12) dự đoán phân phối 16 bins/coordinate → tốt accuracy nhưng **phức tạp cho edge deployment**. YOLO26 bỏ DFL, quay về dự đoán trực tiếp, bù bằng 3 training techniques mạnh hơn [14].

### 6.2.2. MuSGD (Muon-enhanced Stochastic Gradient Descent)

**Vấn đề:** SGD truyền thống hội tụ chậm khi loss landscape phức tạp, dễ bị kẹt ở local minima.

**Giải pháp:** Kết hợp SGD truyền thống với **Muon optimizer** (vốn được phát triển cho training Large Language Models). Muon sử dụng **momentum trên không gian Lie group**, cho phép cập nhật gradient theo hướng tối ưu hơn trên manifold của trọng số [14].

> **Ẩn dụ:** Hãy tưởng tượng bạn đang **leo núi trong sương mù** (tối ưu loss). SGD truyền thống giống bạn chỉ cảm nhận được **độ dốc ngay dưới chân** rồi bước theo hướng xuống — dễ bị kẹt ở thung lũng nhỏ. MuSGD giống bạn có thêm một **la bàn chiến lược** (Muon momentum) nhìn được xu hướng địa hình rộng hơn — khi gặp thung lũng nhỏ, la bàn chỉ bạn bước qua vì phía sau còn có thung lũng sâu hơn. Kết quả: hội tụ nhanh hơn, accuracy cuối cùng cao hơn.

**Hiệu quả:** Thay thế hoàn toàn AdamW (v8–v12) mà accuracy tương đương hoặc cao hơn, sử dụng ít memory hơn — phù hợp edge training [14].

### 6.2.3. STAL (Small-Target-Aware Label Assignment)

**Vấn đề:** Các chiến lược label assignment truyền thống (TAL, SimOTA) ưu tiên vật lớn vì IoU giữa prediction và ground truth tự nhiên cao hơn khi vật lớn. Vật nhỏ (<8 pixels) thường bị gán ít positive samples → model bỏ sót.

**Giải pháp:** STAL điều chỉnh **dynamic** assignment metric dựa trên kích thước ground truth: vật càng nhỏ → threshold gán positive được **nới lỏng** hơn → nhiều predictions được coi là "positive" → model nhận được nhiều training signal hơn cho vật nhỏ [14].

> **Ẩn dụ:** Trong lớp học, **học sinh giỏi** (vật lớn, dễ nhận ra) luôn được thầy khen nhiều — tự nhiên đã có motivation. **Học sinh yếu** (vật nhỏ, khó nhận ra) lại ít được chú ý → ngày càng kém. STAL giống thầy giáo **ưu tiên động viên học sinh yếu**: hạ tiêu chuẩn khen thưởng cho nhóm yếu (nới threshold), để các em nhận được nhiều feedback tích cực hơn → dần cải thiện. Học sinh giỏi thì vẫn giữ tiêu chuẩn cao.

**Hiệu quả:** Đặc biệt tốt cho ứng dụng **drone, camera giám sát tầm xa**, nơi vật thể thường rất nhỏ trong khung hình [14].

### 6.2.4. ProgLoss (Progressive Loss Balancing)

**Vấn đề:** YOLO có 3 thành phần loss (classification, box regression, objectness), cần cân bằng bằng các hệ số λ. Truyền thống dùng hệ số **cố định** suốt training. Nhưng thực tế: giai đoạn đầu model chưa biết gì → cls loss rất cao → chiếm ưu thế → reg loss bị "chìm"; giai đoạn sau model phân loại tốt rồi → cls loss thấp → lẽ ra nên tập trung reg nhưng λ cố định không cho phép.

**Giải pháp:** ProgLoss **tự động điều chỉnh** hệ số cân bằng $\lambda_i$ theo training progress (epoch hiện tại / tổng epochs). Giai đoạn đầu: ưu tiên cls (cho model học nhận diện trước). Giai đoạn sau: dần chuyển trọng tâm sang reg (tinh chỉnh vị trí) [14].

> **Ẩn dụ:** Giống quá trình **học lái xe**. Tuần đầu (epoch sớm), giáo viên tập trung dạy **nhận biết biển báo, đèn giao thông** trước (classification — biết cái gì). Tuần sau, khi bạn đã biết biển báo, giáo viên chuyển sang dạy **căn đường, đỗ xe chính xác** (regression — ở đâu, chính xác cỡ nào). Nếu giáo viên cứ dạy song song cả hai từ đầu với cùng tỷ lệ (λ cố định), bạn sẽ bị quá tải và không giỏi cái nào cả.

**Hiệu quả:** +0.3–0.5% mAP so với fixed loss weights, đặc biệt hiệu quả khi bỏ DFL (cần reg loss mạnh hơn để bù đắp) [14].

### 6.2.5. Kết quả

| Model | Params (ước tính) | mAP@50:95 | CPU Speed | So với YOLO11 |
|:--|:-:|:-:|:--|:--|
| YOLO26-N | ~2.5M | ~40.5% | Nhanh hơn 43% | +1.0% |
| YOLO26-S | ~9M | ~47.5% | Nhanh hơn 30% | +0.5% |
| YOLO26-X | ~60M | ~57.2% | Nhanh hơn 15% | +2.5% |

*Ghi chú: Số liệu ước tính từ benchmark cộng đồng, chưa có paper chính thức. CPU speed đo trên ONNX Runtime. YOLO26-X (~57.2%) vượt kỷ lục mAP của v7-E6E (56.8%) — lần đầu tiên sau 4 năm [7][14].*

### 6.2.6. Tổng kết triết lý YOLO26

YOLO26 đại diện cho xu hướng **"less is more"**: bỏ bớt component phức tạp (DFL), thay bằng training techniques thông minh hơn (MuSGD, STAL, ProgLoss). Kết quả: model **nhẹ hơn, nhanh hơn trên CPU/edge**, mà accuracy vẫn SOTA.
