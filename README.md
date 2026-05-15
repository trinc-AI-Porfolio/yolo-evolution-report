# 📚 YOLO Evolution Report — Từ YOLOv1 đến YOLO26

> Khảo sát toàn diện sự phát triển của YOLO (You Only Look Once) qua 11 năm (2015–2026), 14+ phiên bản.

---

## 📂 Cấu trúc thư mục

```
📦 yolo-evolution-report
├── 📄 README.md
├── 📁 baocao_yolo_v2/          ← Báo cáo LaTeX chính thức (41 trang)
│   ├── Report_Yolo_Version.pdf ← 📥 TẢI PDF TẠI ĐÂY
│   ├── main.tex                ← Source LaTeX
│   └── *.png                   ← Biểu đồ minh họa
├── 📁 Yolo_Version/            ← Tài liệu chi tiết từng phiên bản
│   ├── yolov1.md → yolov12.md
│   ├── yolo11.md, yolo26.md
│   └── yoloe.md
├── 📁 Yolo_Bien_The/           ← Các biến thể YOLO
│   └── yolo_bien_the.md
├── 📄 so_sanh_version.md       ← So sánh đa chiều giữa các version
└── 📄 uu_diem_va_nhuoc_diem_cac_version.md ← Ưu/nhược điểm từng version
```

---

## 📥 Báo cáo chính

**[📄 Report_Yolo_Version.pdf](baocao_yolo_v2/Report_Yolo_Version.pdf)** — Báo cáo LaTeX hoàn chỉnh (41 trang, 9 chương)


## 📁 Yolo_Version — Chi tiết từng phiên bản

| File | Phiên bản | Điểm nhấn |
|:--|:--|:--|
| [`yolov1.md`].(baocao_yolo_v2/yolov1.md) | YOLOv1 (2015) | Khởi nguồn Single-stage Detection |
| `yolov2.md` | YOLOv2 (2016) | Anchor boxes, Batch Normalization |
| `yolov3.md` | YOLOv3 (2018) | Multi-Scale Detection (FPN) |
| `yolov4.md` | YOLOv4 (2020) | Gộp những gì tốt nhất BoF/BoS, CIoU, Mosaic |
| `yolov5.md` | YOLOv5 (2020) | PyTorch, pip install|
| `yolov6.md` | YOLOv6 (2022) | Industrial Deployment (Re-param, Quantization) |
| `yolov7.md` | YOLOv7 (2022) | SOTA Accuracy (E-ELAN, 56.8% mAP) |
| `yolov8.md` | YOLOv8 (2023) | Đa năng nhất (Anchor-free, 6 tasks) |
| `yolov9.md` | YOLOv9 (2024) | PGI & Information Bottleneck |
| `yolov10.md` | YOLOv10 (2024) | NMS-Free, hiệu quả params nhất |
| `yolo11.md` | YOLO11 (2024) | C3K2, C2PSA |
| `yolov12.md` | YOLOv12 (2025) | Attention-Centric (R-ELAN, Area Attention) |
| `yolo26.md` | YOLO26 (2026) | Edge-First Design (MuSGD, STAL, No DFL) |
| `yoloe.md` | YOLOE (2025) | Open-Vocabulary Real-Time Detection |

Note: NMS (Non-Maximum Suppression) là một thuật toán hậu xử lý (post-processing)
---

## 📊 Tài liệu so sánh & phân tích

| File | Nội dung |
|:--|:--|
| [`so_sanh_version.md`](so_sanh_version.md) | So sánh đa chiều: mAP, FPS, Params, Ecosystem |
| [`uu_diem_va_nhuoc_diem_cac_version.md`](uu_diem_va_nhuoc_diem_cac_version.md) | Ưu điểm & nhược điểm chi tiết từng version |
| [`Yolo_Bien_The/yolo_bien_the.md`](Yolo_Bien_The/yolo_bien_the.md) | Các biến thể: YOLOX, PP-YOLOE, YOLO-NAS, YOLO-World |

---

## 👤 Tác giả

**Nguyễn Cao Trị**

## 📜 License

Tài liệu này được tạo cho mục đích học tập và nghiên cứu.
