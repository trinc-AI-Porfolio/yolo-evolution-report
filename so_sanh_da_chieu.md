# SO SÁNH ĐA CHIỀU GIỮA CÁC ĐỜI YOLO

*Phân tích từ nhiều góc độ: accuracy, tốc độ, kiến trúc, ecosystem, và liệu đời sau có thật sự tốt hơn đời trước?*

---

## 1. BẢNG SO SÁNH TỔNG QUAN

| Version | Năm | mAP (best) | Params (best) | FPS | Anchor | NMS | Multi-Task | Paper? |
|---------|-----|-----------|--------------|-----|:------:|:---:|:----------:|:------:|
| v1 | 2015 | 63.4% (VOC) | ~45M | 45 | ❌ | Cần | ❌ | ✅ |
| v2 | 2016 | 78.6% (VOC) | ~50M | 40-67 | ✅ | Cần | ❌ | ✅ |
| v3 | 2018 | 33.0% (COCO) | ~62M | 20-45 | ✅ | Cần | ❌ | ✅ |
| v4 | 2020 | 43.5% | ~64M | 65 | ✅ | Cần | ❌ | ✅ |
| v5 | 2020 | 50.7% | 86.7M | 45-140 | ✅ | Cần | ⚠️ | ❌ |
| v6 | 2022 | 52.5% | ~58M | 520* | ❌ | Cần | ❌ | ✅ |
| v7 | 2022 | 56.8% | ~71M | 36-161 | ✅ | Cần | ⚠️ | ✅ |
| v8 | 2023 | 53.9% | 68.2M | 40-280 | ❌ | Cần | ✅✅ | ❌ |
| v9 | 2024 | 55.6% | 58.1M | 30-115 | ❌ | Cần | ⚠️ | ✅ |
| v10 | 2024 | 54.4% | 29.5M | 50-300 | ❌ | ❌ | ❌ | ✅ |
| v11 | 2024 | 54.7% | 56.9M | 50-280 | ❌ | ❌ | ✅✅ | ❌ |
| v12 | 2025 | 55.2% | 59.1M | 40-200 | ❌ | ❌ | ✅ | ✅ |
| YOLOE | 2025 | 27%* | ~25M | RT | ❌ | Tùy | ✅ | ✅ |
| YOLO26 | 2026 | 57.2% | ~60M | 50-300 | ❌ | ❌ | ✅✅ | ❌ |

*\* v6 FPS đo trên TensorRT FP16, khác điều kiện. YOLOE đo trên LVIS zero-shot (thang đo khác).*

---

## 2. SO SÁNH TỪNG CẶP: ĐỜI SAU CÓ THẬT SỰ TỐT HƠN ĐỜI TRƯỚC?

### v1 → v2: ✅ Tốt hơn rõ ràng
| Góc nhìn | v1 | v2 | Nhận xét |
|----------|----|----|---------|
| Accuracy | 63.4% (VOC) | 78.6% (VOC) | v2 thắng tuyệt đối (+15%) |
| Recall | 81% | 88% | Anchor boxes giúp nhiều |
| Vật nhỏ | Rất yếu | Tốt hơn (passthrough) | Cải thiện nhưng vẫn chưa đủ |
| Tốc độ | 45 FPS | 40-67 FPS | Tương đương |
| Kiến trúc | Thô sơ | Darknet-19, anchor | Bước tiến lớn |
**Kết luận: v2 tốt hơn v1 mọi mặt. Không có trade-off.**

### v2 → v3: ✅ Tốt hơn nhưng có trade-off
| Góc nhìn | v2 | v3 | Nhận xét |
|----------|----|----|---------|
| Accuracy mAP50 | ~44% (COCO) | 57.9% | v3 thắng lớn |
| Accuracy mAP50-95 | ~21% | 33.0% | v3 thắng |
| Backbone | Darknet-19 | Darknet-53 | Mạnh hơn nhưng nặng hơn |
| Tốc độ | 67 FPS | 20-51 FPS | **v3 chậm hơn** ⚠️ |
| Multi-scale | 1 scale | 3 scales | Đột phá |
**Kết luận: v3 chính xác hơn nhiều, nhưng chậm hơn v2. Trade-off tốc độ lấy accuracy.**

### v3 → v4: ✅ Tốt hơn mọi mặt
| Góc nhìn | v3 | v4 | Nhận xét |
|----------|----|----|---------|
| mAP50-95 | 33.0% | 43.5% | +10.5% — nhảy vọt lớn nhất lịch sử YOLO |
| Tốc độ | ~20 FPS (608) | ~65 FPS | v4 nhanh hơn + chính xác hơn |
| Augmentation | Cơ bản | Mosaic, CutMix, SAT | Bước tiến lớn |
| Loss | MSE-based | CIoU | Localization tốt hơn |
| Framework | Darknet C | Darknet C | Vẫn khó dùng |
**Kết luận: v4 vượt v3 hoàn toàn. Bước nhảy lớn nhất trong lịch sử YOLO.**

### v4 → v5: ⚠️ Không rõ ràng — Phụ thuộc góc nhìn
| Góc nhìn | v4 | v5 | Ai thắng? |
|----------|----|----|----------|
| Accuracy | 43.5% | 50.7% | v5 (nhưng so sánh không công bằng — v5 có nhiều sizes) |
| Tốc độ | 65 FPS | Đa dạng (1.9M→86.7M) | v5 linh hoạt hơn |
| Dễ dùng | ⭐⭐ (Darknet C) | ⭐⭐⭐⭐⭐ (pip install) | **v5 thắng tuyệt đối** |
| Đóng góp kỹ thuật | CIoU, Mosaic, BoF/BoS | Không đột phá mới | **v4 thắng** |
| Paper | Có (chi tiết) | Không | **v4 thắng** |
| Community | Nhỏ | Rất lớn | v5 thắng |
**Kết luận: v5 dễ dùng hơn RẤT NHIỀU nhưng không đóng góp kỹ thuật mới. v4 đóng góp nhiều hơn cho lĩnh vực. Tùy bạn cần gì: dùng ngay → v5, nghiên cứu → v4.**

### v5 → v6: ⚠️ Không rõ ràng
| Góc nhìn | v5 | v6 | Ai thắng? |
|----------|----|----|----------|
| Accuracy | 50.7% | 52.5% | v6 nhỉnh |
| Tốc độ | Tốt | 520 FPS (TensorRT!) | v6 (nhưng đo khác điều kiện) |
| Ecosystem | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | **v5 thắng** |
| Kỹ thuật mới | Ít | Re-param, TAL, Quantization | **v6 thắng** |
| Community | Rất lớn | Nhỏ (Meituan nội bộ) | **v5 thắng** |
**Kết luận: v6 kỹ thuật tốt hơn nhưng v5 dễ dùng hơn. Đa số người dùng vẫn chọn v5.**

### v6 → v7: ✅ v7 tốt hơn về accuracy
| Góc nhìn | v6 | v7 | Nhận xét |
|----------|----|----|---------|
| Accuracy | 52.5% | 56.8% | v7 thắng lớn |
| Tốc độ | 520 FPS (TRT) | 36-161 FPS | **v6 nhanh hơn** (nhưng đo khác) |
| Kiến trúc | EfficientRep | E-ELAN | Cả 2 đều sáng tạo |
| Dễ dùng | ⭐⭐⭐ | ⭐⭐⭐ | Ngang nhau, đều khó hơn v5 |
**Kết luận: v7 SOTA accuracy, v6 SOTA industrial speed. Khác mục đích.**

### v7 → v8: ⚠️ PHỨC TẠP NHẤT — v8 thua v7 về accuracy!
| Góc nhìn | v7 | v8 | Ai thắng? |
|----------|----|----|----------|
| **mAP (best)** | **56.8%** | **53.9%** | **v7 thắng!** ⚠️ |
| Ecosystem | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **v8 thắng** |
| Multi-task | ⚠️ Det only | ✅ 6 tác vụ | **v8 thắng** |
| Anchor-free | ❌ (anchor) | ✅ | v8 hiện đại hơn |
| DFL | ❌ | ✅ | v8 localization tốt hơn |
| Dễ dùng | Khó | Rất dễ | **v8 thắng** |
| Paper | Có (CVPR) | Không | **v7 thắng** |

**🔴 v8 THUA v7 về accuracy thuần túy!** v7-E6E (56.8%) > v8-X (53.9%). Tuy nhiên:
- So sánh không hoàn toàn công bằng: v7-E6E là model rất lớn, chạy 36 FPS
- v8 thắng ở ecosystem, multi-task, dễ dùng
- **Nếu cần accuracy tối đa → v7. Nếu cần đa năng + dễ dùng → v8.**

### v8 → v9: ✅ v9 tốt hơn về accuracy, kém ecosystem
| Góc nhìn | v8 | v9 | Nhận xét |
|----------|----|----|---------|
| Accuracy | 53.9% | 55.6% | v9 thắng |
| Params | 68.2M (X) | 58.1M (E) | v9 hiệu quả hơn |
| Đóng góp lý thuyết | Ít | PGI — rất sâu sắc | **v9 thắng** |
| Ecosystem | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | **v8 thắng** |
| Multi-task | ✅ 6 tác vụ | ⚠️ Hạn chế | **v8 thắng** |
**Kết luận: v9 mạnh hơn kỹ thuật nhưng v8 hoàn thiện hơn cho production.**

### v9 → v10: ⚠️ Trade-off rõ ràng
| Góc nhìn | v9 | v10 | Nhận xét |
|----------|----|----|---------|
| **mAP (best)** | **55.6%** | **54.4%** | **v9 thắng!** ⚠️ |
| Params | 58.1M | 29.5M | **v10 thắng** (gần 2× ít hơn) |
| NMS | Cần | Không cần | **v10 thắng** |
| Multi-task | ⚠️ | ❌ Det only | **v9 thắng** |
| Efficiency | Tốt | Rất tốt | v10 hiệu quả hơn |

**🔴 v10 THUA v9 về accuracy!** v10-X (54.4%) < v9-E (55.6%). Nhưng v10 ít params hơn gần 2× và không cần NMS. **v10 đổi accuracy lấy efficiency.**

### v10 → v11: ✅ v11 tốt hơn hầu hết mặt
| Góc nhìn | v10 | v11 | Nhận xét |
|----------|----|----|---------|
| Accuracy | 54.4% | 54.7% | v11 nhỉnh |
| Multi-task | ❌ Det only | ✅ 6 tác vụ | **v11 thắng lớn** |
| Ecosystem | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **v11 thắng** |
| NMS-free | ✅ | ✅ | Ngang |
| Attention | PSA (partial) | C2PSA | v11 phát triển hơn |
**Kết luận: v11 tốt hơn v10 hầu hết mặt. Đời sau thắng đời trước.**

### v11 → v12: ⚠️ Trade-off
| Góc nhìn | v11 | v12 | Nhận xét |
|----------|----|----|---------|
| mAP (best) | 54.7% | 55.2% | v12 nhỉnh |
| Kiến trúc | CNN-centric | Attention-centric | v12 hiện đại hơn |
| Hardware yêu cầu | GPU thường | Cần FlashAttention | **v11 thắng** |
| Ecosystem | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | v11 nhỉnh |
| Đóng góp lý thuyết | Ít | Area Attention, R-ELAN | **v12 thắng** |
**Kết luận: v12 accuracy nhỉnh hơn nhưng cần GPU hiện đại hơn.**

### v12 → YOLO26: ✅ YOLO26 tốt hơn mọi mặt
| Góc nhìn | v12 | YOLO26 | Nhận xét |
|----------|----|----|---------|
| mAP | 55.2% | 57.2% | **YOLO26 thắng (+2%)** |
| Tốc độ CPU | Cơ bản | Nhanh hơn YOLO11 43% | **YOLO26 thắng** |
| Vật nhỏ | Bình thường | STAL chuyên vật <8px | **YOLO26 thắng** |
| Maturity | 4 tháng | Mới nhất | **v12 thắng** |
**Kết luận: YOLO26 vượt v12 nhưng còn quá mới.**

---

## 3. CÁC VERSION BỊ THUA BỞI VERSION TRƯỚC

| Trường hợp | Thua ở | Nhưng thắng ở |
|-----------|--------|-------------|
| **v8 thua v7** | mAP: 53.9% < 56.8% | Ecosystem, multi-task, dễ dùng |
| **v10 thua v9** | mAP: 54.4% < 55.6% | Params ít 2×, NMS-free |
| **v3 chậm hơn v2** | FPS: 20-51 < 40-67 | Accuracy tốt hơn nhiều |

**Nhận xét:** Không có version nào thua hoàn toàn. Mỗi version "thua" ở 1 khía cạnh nhưng "thắng" ở khía cạnh khác → trade-off có chủ đích.

---

## 4. SO SÁNH THEO TIÊU CHÍ CỤ THỂ

### 4.1. Accuracy thuần túy (mAP50-95 COCO, best model)
```
v1 (VOC) → v2 (VOC) → v3 (33%) → v4 (43.5%) → v5 (50.7%) → v6 (52.5%)
→ v7 (56.8%) → v8 (53.9%) ← GIẢM! → v9 (55.6%) → v10 (54.4%) ← GIẢM!
→ v11 (54.7%) → v12 (55.2%) → YOLO26 (57.2%)
```
**Đỉnh cao accuracy:** v7 (56.8%) cho đến khi YOLO26 vượt qua (57.2%).

### 4.2. Efficiency (mAP / Params)
| Version | mAP | Params | mAP/M params |
|---------|-----|--------|:------------:|
| v4 | 43.5% | ~64M | 0.68 |
| v8-X | 53.9% | 68.2M | 0.79 |
| v9-E | 55.6% | 58.1M | 0.96 |
| **v10-X** | **54.4%** | **29.5M** | **1.84** ← Hiệu quả nhất! |
| v11-X | 54.7% | 56.9M | 0.96 |

**v10 hiệu quả nhất** — cùng accuracy nhưng ít params nhất.

### 4.3. Dễ dùng (Ecosystem)
```
Rất khó: v1, v2, v3 (Darknet C, compile thủ công)
Khó: v4 (Darknet C, config phức tạp)
Trung bình: v6 (Meituan), v7 (code riêng), v9 (hệ sinh thái nhỏ)
Dễ: v12 (Ultralytics nhưng mới), YOLOE, YOLO26
Rất dễ: v5, v8, v11 (pip install, 1 dòng code train)
```

### 4.4. Đóng góp lý thuyết
| Hạng | Version | Đóng góp |
|:----:|---------|---------|
| 🥇 | v1 | Single-stage detection — thay đổi cả lĩnh vực |
| 🥈 | v9 | Information Bottleneck + PGI — lý thuyết sâu nhất |
| 🥉 | v4 | BoF/BoS methodology — hệ thống hóa kỹ thuật |
| 4 | v10 | NMS-free — giải quyết vấn đề 9 năm |
| 5 | v12 | Attention-centric — mở kỷ nguyên mới |
| 6 | v3 | Multi-scale prediction — giải quyết vật nhỏ |
| 7 | YOLOE | Open-vocabulary real-time |
| 8 | v7 | E-ELAN + planned re-param |
| 9 | v2 | Anchor boxes, BN, WordTree |

### 4.5. Phù hợp production (2026)
| Hạng | Version | Lý do |
|:----:|---------|-------|
| 🥇 | YOLO11 | Đa năng nhất, ổn định, ecosystem tốt |
| 🥈 | YOLOv8 | Mature nhất, ít bug, nhiều tutorial |
| 🥉 | YOLO26 | mAP cao nhất, nhưng còn mới |
| 4 | YOLOv12 | Attention mạnh, nhưng cần GPU tốt |

---

## 5. MỐI QUAN HỆ ẢNH HƯỞNG

```
v1 (Redmon) → v2 (Redmon) → v3 (Redmon) ──→ DỪNG
                                    ↓
                              v4 (Bochkovskiy + Wang)
                              ↓            ↓
                         v5 (Ultralytics)  v7 (Wang) ──→ v9 (Wang)
                              ↓
                         v8 (Ultralytics) → v11 (Ultralytics) → YOLO26 (Ultralytics)
                              ↓
                         v10 (Tsinghua) → v12 (Tian)

Nhánh phụ:
  YOLOX (Megvii) ──→ ảnh hưởng v6, v8 (anchor-free, decoupled head)
  PP-YOLOE (Baidu) → ảnh hưởng v6 (TAL)
  YOLO-World (Tencent) → YOLOE (open-vocab)
  v6 (Meituan) → ảnh hưởng v8 (re-param, anchor-free)
```

---

## 6. KẾT LUẬN

**Không có "version tốt nhất tuyệt đối"** — mỗi version tối ưu cho 1 mục đích khác nhau:

| Bạn cần gì? | Chọn | Lý do |
|-------------|------|-------|
| Accuracy tối đa | YOLO26 hoặc v7 | mAP cao nhất |
| Dễ dùng nhất | v8 hoặc v11 | pip install, tutorials |
| Ít params nhất | v10 | 29.5M cho 54.4% |
| NMS-free | v10, v11, YOLO26 | End-to-end pipeline |
| Detect vật chưa train | YOLOE | Open-vocabulary |
| Nghiên cứu/học tập | v1 → v4 papers | Nền tảng lý thuyết |
| Industrial/edge | v6, YOLO26 | Tối ưu deployment |

**Sự thật cuối cùng:** Chênh lệch 1-3% mAP giữa các version trên COCO **có thể không có ý nghĩa** trên dataset thực tế của bạn. Data chất lượng quan trọng hơn chọn model.

---

*Tài liệu so sánh đa chiều — cập nhật 05/2026*
