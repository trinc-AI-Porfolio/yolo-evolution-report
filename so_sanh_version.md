# SO SÁNH ĐA CHIỀU GIỮA CÁC ĐỜI YOLO

*Phân tích từ nhiều góc độ: accuracy, tốc độ, kiến trúc, ecosystem, và liệu đời sau có thật sự tốt hơn đời trước?*

---

## 1. BẢNG SO SÁNH TỔNG QUAN

---

| Version | Năm | mAP (best) | Params (best) | FPS | Anchor | NMS | Multi-Task | Paper? |
|---------|-----|-----------|--------------|-----|:------:|:---:|:----------:|:------:|
| v1 | 2015 | 63.4% (VOC) | ~45M | 45 | ❌ | Cần | ❌ | ⭕ |
| v2 | 2016 | 78.6% (VOC) | ~50M | 40-67 | ⭕ | Cần | ❌ | ⭕ |
| v3 | 2018 | 33.0% (COCO) | ~62M | 20-45 | ⭕ | Cần | ❌ | ⭕ |
| v4 | 2020 | 43.5% (COCO) | ~64M | 65 | ⭕ | Cần | ❌ | ⭕ |
| v5 | 2020 | 50.7% (COCO) | 86.7M | 45-140 | ⭕ | Cần | ⚠️ | ❌ |
| v6 | 2022 | 52.5% (COCO) | ~58M | 520* | ❌ | Cần | ❌ | ⭕ |
| v7 | 2022 | 56.8% (COCO) | ~71M | 36-161 | ⭕ | Cần | ⚠️ | ⭕ |
| v8 | 2023 | 53.9% (COCO) | 68.2M | 40-280 | ❌ | Cần | ⭕⭕ | ❌ |
| v9 | 2024 | 55.6% (COCO) | 58.1M | 30-115 | ❌ | Cần | ⚠️ | ⭕ |
| v10 | 2024 | 54.4% (COCO) | 29.5M | 50-300 | ❌ | ❌ | ❌ | ⭕ |
| v11 | 2024 | 54.7% (COCO) | 56.9M | 50-280 | ❌ | ❌ | ⭕⭕ | ❌ |
| v12 | 2025 | 55.2% (COCO) | 59.1M | 40-200 | ❌ | ❌ | ⭕ | ⭕ |
| YOLOE | 2025 | 27% (LVIS)* | ~25M | RT | ❌ | Tùy | ⭕ | ⭕ |
| YOLO26 | 2026 | 57.2% (COCO) | ~60M | 50-300 | ❌ | ❌ | ⭕⭕ | ❌ |

*\* v6 FPS đo trên TensorRT FP16, khác điều kiện. YOLOE đo trên LVIS zero-shot (thang đo khác).*
*\* NMS (Non-Maximum Suppression) là một thuật toán hậu xử lý (post-processing).*
---

## 2. SO SÁNH TỪNG CẶP: ĐỜI SAU CÓ THẬT SỰ TỐT HƠN ĐỜI TRƯỚC?

### v1 → v2:  Tốt hơn rõ ràng
---
| Góc nhìn | v1 | v2 | Nhận xét |
|----------|----|----|---------|
| Accuracy | 63.4% (VOC) | 78.6% (VOC) | v2 tốt hơn (+15%) |
| Recall | 81% | 88% | Anchor boxes giúp nhiều |
| Vật nhỏ | Rất yếu | Tốt hơn (passthrough) | Cải thiện nhưng vẫn chưa đủ |
| Tốc độ | 45 FPS | 40-67 FPS | Tương đương |
| Kiến trúc | Thô sơ | Darknet-19, anchor | Bước tiến lớn |

**Kết luận: v2 tốt hơn v1 mọi mặt. Không có trade-off.**

### v2 → v3: Tốt hơn nhưng có trade-off
---
| Góc nhìn | v2 | v3 | Nhận xét |
|----------|----|----|---------|
| Accuracy mAP50 | ~44% (COCO) | 57.9% | v3 tốt hơn|
| Accuracy mAP50-95 | ~21% | 33.0% | v3 tốt hơn |
| Backbone | Darknet-19 | Darknet-53 | Mạnh hơn nhưng nặng hơn |
| Tốc độ | 67 FPS | 20-51 FPS | **v3 chậm hơn** ⚠️ |
| Multi-scale | 1 scale | 3 scales | Đột phá |

**Kết luận: v3 chính xác hơn nhiều, nhưng chậm hơn v2. Trade-off tốc độ lấy accuracy.**

### v3 → v4: Tốt hơn mọi mặt
---
| Góc nhìn | v3 | v4 | Nhận xét |
|----------|----|----|---------|
| mAP50-95 | 33.0% | 43.5% | +10.5% — nhảy vọt lớn nhất lịch sử YOLO |
| Tốc độ | ~20 FPS (608) | ~65 FPS | v4 nhanh hơn + chính xác hơn |
| Augmentation | Cơ bản | Mosaic, CutMix, SAT | Bước tiến lớn |
| Loss | MSE-based | CIoU | Localization tốt hơn |
| Framework | Darknet C | Darknet C | Vẫn khó dùng |
**Kết luận: v4 tốt hơn v3 hoàn toàn. Bước nhảy lớn nhất trong lịch sử YOLO.**

### v4 → v5: Không rõ ràng — Phụ thuộc góc nhìn
---
| Góc nhìn | v4 | v5 | Nhận xét |
|----------|----|----|----------|
| Accuracy | 43.5% | 50.7% | v5 (nhưng so sánh không công bằng — v5 có nhiều sizes) |
| Tốc độ | 65 FPS | Đa dạng (1.9M→86.7M) | v5 linh hoạt hơn |
| Dễ dùng | Darknet C | pip install)| **v5 tốt hơn** |
| Đóng góp kỹ thuật | CIoU, Mosaic, BoF/BoS | Không đột phá mới | **v4 tốt hơn** |
| Paper | Có (chi tiết) | Không | **v4 tốt hơn** |
| Community | Nhỏ | Rất lớn | v5 tốt hơn |

**Kết luận: v5 dễ dùng hơn RẤT NHIỀU nhưng không đóng góp kỹ thuật mới. v4 đóng góp nhiều hơn cho lĩnh vực. Tùy bạn cần gì: dùng ngay → v5, nghiên cứu → v4.**

### v5 → v6: Không rõ ràng
---
| Góc nhìn | v5 | v6 | Nhận xét |
|----------|----|----|----------|
| Accuracy | 50.7% | 52.5% | v6 nhỉnh hơn |
| Tốc độ | Rất nhanh | 520 FPS (TensorRT!) | v6 (nhưng đo khác điều kiện) |
| Ecosystem | Rất tốt | Trung bình | **v5 tốt hơn** |
| Kỹ thuật mới | Ít | Re-param, TAL, Quantization | **v6 tốt hơn** |
| Community | Rất lớn | Nhỏ (Meituan nội bộ) | **v5 tốt hơn** |

**Kết luận: v6 kỹ thuật tốt hơn nhưng v5 dễ dùng hơn. Đa số người dùng vẫn chọn v5.**

### v6 → v7: v7 tốt hơn về accuracy
---
| Góc nhìn | v6 | v7 | Nhận xét |
|----------|----|----|---------|
| Accuracy | 52.5% | 56.8% | v7 tốt hơn |
| Tốc độ | 520 FPS (TRT) | 36-161 FPS | **v6 nhanh hơn** (nhưng đo khác) |
| Kiến trúc | EfficientRep | E-ELAN | Cả 2 đều sáng tạo |
| Dễ dùng | Trung Bình | Trung Bình | Ngang nhau, đều khó hơn v5 |

**Kết luận: v7 SOTA accuracy, v6 SOTA industrial speed. Khác mục đích.**

### v7 → v8: ⚠️ PHỨC TẠP NHẤT — v8 tệ hơn v7 về accuracy!
---
| Góc nhìn | v7 | v8 | Nhận xét |
|----------|----|----|----------|
| **mAP (best)** | **56.8%** | **53.9%** | **v7 tốt hơn!** ⚠️ |
| Ecosystem | Trung Bình | Rất tốt | **v8 tốt hơn** |
| Multi-task | ⚠️ Det only | ✅ 6 tác vụ | **v8 tốt hơn** |
| Anchor-free | ❌ (anchor) | ✅ | v8 hiện đại hơn |
| DFL | ❌ | ✅ | v8 localization tốt hơn |
| Dễ dùng | Khó | Rất dễ | **v8 tốt hơn** |
| Paper | Có (CVPR) | Không | **v7 tốt hơn** |

**🔴 v8 tệ hơn v7 về accuracy thuần túy!** v7-E6E (56.8%) > v8-X (53.9%). Tuy nhiên:
- So sánh không hoàn toàn công bằng: v7-E6E là model rất lớn, chạy 36 FPS
- v8 tốt hơn về ecosystem, multi-task, dễ dùng
- **Nếu cần accuracy tối đa → v7. Nếu cần đa năng + dễ dùng → v8.**

### v8 → v9: v9 tốt hơn về accuracy, kém ecosystem
---
| Góc nhìn | v8 | v9 | Nhận xét |
|----------|----|----|---------|
| Accuracy | 53.9% | 55.6% | v9 tốt hơn |
| Params | 68.2M (X) | 58.1M (E) | v9 hiệu quả hơn |
| Đóng góp lý thuyết | Ít | PGI — rất sâu sắc | **v9 tốt hơn** |
| Ecosystem | Rất tốt | Trung bình | **v8 tốt hơn** |
| Multi-task | ⭕ 6 tác vụ | ⚠️ Hạn chế | **v8 tốt hơn** |

**Kết luận: v9 mạnh hơn kỹ thuật nhưng v8 hoàn thiện hơn cho production.**

### v9 → v10: Trade-off rõ ràng
---
| Góc nhìn | v9 | v10 | Nhận xét |
|----------|----|----|---------|
| **mAP (best)** | **55.6%** | **54.4%** | **v9 tốt hơn!** ⚠️ |
| Params | 58.1M | 29.5M | **v10 tốt hơn** (gần 2× ít hơn) |
| NMS | Cần | Không cần | **v10 tốt hơn** |
| Multi-task | ⚠️ | ❌ Det only | **v9 tốt hơn** |
| Efficiency | Tốt | Rất tốt | v10 tốt hơn |

**🔴 v10 tệ hơn v9 về accuracy!** v10-X (54.4%) < v9-E (55.6%). Nhưng v10 ít params hơn gần 2× và không cần NMS. **v10 đổi accuracy lấy efficiency.**

### v10 → v11: v11 tốt hơn hầu hết mặt
---
| Góc nhìn | v10 | v11 | Nhận xét |
|----------|----|----|---------|
| Accuracy | 54.4% | 54.7% | v11 tốt hơn |
| Multi-task | ❌ Det only | ✅ 6 tác vụ | **v11 tốt hơn** |
| Ecosystem | Trung bình | Rất tốt | **v11 tốt hơn** |
| NMS-free | ⭕ | ⭕ | Ngang |
| Attention | PSA (partial) | C2PSA | v11 phát triển hơn |

**Kết luận: v11 tốt hơn v10 hầu hết mặt. Đời sau tốt hơn đời trước.**

### v11 → v12:  Trade-off
---
| Góc nhìn | v11 | v12 | Nhận xét |
|----------|----|----|---------|
| mAP (best) | 54.7% | 55.2% | v12 tốt hơn |
| Kiến trúc | CNN-centric | Attention-centric | v12 hiện đại hơn |
| Hardware yêu cầu | GPU thường | Cần FlashAttention | **v11 tốt hơn** |
| Ecosystem | Rất tốt | tốt | v11 tốt hơn |
| Đóng góp lý thuyết | Ít | Area Attention, R-ELAN | **v12 tốt hơn** |

**Kết luận: v12 accuracy nhỉnh hơn nhưng cần GPU hiện đại hơn.**

### v12 → YOLO26:  YOLO26 tốt hơn mọi mặt
---
| Góc nhìn | v12 | YOLO26 | Nhận xét |
|----------|----|----|---------|
| mAP | 55.2% | 57.2% | **YOLO26 tốt hơn (+2%)** |
| Tốc độ CPU | Cơ bản | Nhanh hơn YOLO11 43% | **YOLO26 tốt hơn** |
| Vật nhỏ | Bình thường | STAL chuyên vật <8px | **YOLO26 tốt hơn** |
| Maturity | 4 tháng | Mới nhất | **v12 tốt hơn** |

**Kết luận: YOLO26 vượt v12 nhưng còn quá mới.**

---

## 3. SO SÁNH THEO TIÊU CHÍ CỤ THỂ

### 3.1. Accuracy thuần túy (mAP50-95 COCO, best model)
```
v1 (VOC) → v2 (VOC) → v3 (33%) → v4 (43.5%) → v5 (50.7%) → v6 (52.5%)
→ v7 (56.8%) → v8 (53.9%) ↓ → v9 (55.6%) → v10 (54.4%) ↓
→ v11 (54.7%) → v12 (55.2%) → 𝐘𝐎𝐋𝐎𝟐𝟔 (𝟓𝟕.𝟐%)
```

### 3.2. Efficiency (mAP / Params)
| Version | mAP | Params | mAP/M params |
|---------|-----|--------|:------------:|
| v4 | 43.5% | ~64M | 0.68 |
| v8-X | 53.9% | 68.2M | 0.79 |
| v9-E | 55.6% | 58.1M | 0.96 |
| **v10-X** | **54.4%** | **29.5M** | **1.84** ← Hiệu quả nhất! |
| v11-X | 54.7% | 56.9M | 0.96 |



### 3.3. Dễ dùng (Ecosystem)
```
Rất khó: v1, v2, v3 (Darknet C, compile thủ công)
Khó: v4 (Darknet C, config phức tạp)
Trung bình: v6 (Meituan), v7 (code riêng), v9 (hệ sinh thái nhỏ)
Dễ: v12 (Ultralytics nhưng mới), YOLOE, YOLO26
Rất dễ: v5, v8, v11 (pip install, 1 dòng code train)
```

### 3.4. Đóng góp lý thuyết
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

### 3.5. Phù hợp production (2026)
| Hạng | Version | Lý do |
|:----:|---------|-------|
| 🥇 | YOLO11 | Đa năng nhất, ổn định, ecosystem tốt |
| 🥈 | YOLOv8 | Mature nhất, ít bug, nhiều tutorial |
| 🥉 | YOLO26 | mAP cao nhất, nhưng còn mới |
| 4 | YOLOv12 | Attention mạnh, nhưng cần GPU tốt |

---

## 4. KẾT LUẬN

**Không có "version tốt nhất tuyệt đối"** — mỗi version tối ưu cho 1 mục đích khác nhau:
---
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
