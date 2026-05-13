# 📚 Mastering The YOLO Ecosystem — Từ YOLOv1 đến YOLO26

> Tài liệu nghiên cứu toàn diện về sự phát triển của YOLO (You Only Look Once) qua 11 năm (2015–2026).

## 📖 Nội dung

### 📄 Báo cáo chính
- [`baocao_yolo_v2/main.pdf`](baocao_yolo_v2/main.pdf) — **Báo cáo LaTeX hoàn chỉnh** (48 trang, 10 chương)

### 📝 Tài liệu dịch & phân tích từng version
| File | Nội dung |
|:--|:--|
| `dich_yolov1.md` | YOLOv1 — Khởi nguồn Single-stage Detection |
| `dich_yolov2.md` | YOLOv2/YOLO9000 — Better, Faster, Stronger |
| `dich_yolov3.md` | YOLOv3 — Multi-Scale Detection |
| `dich_yolov4.md` | YOLOv4 — Tập đại thành (BoF/BoS) |
| `dich_yolov5.md` | YOLOv5 — Dân chủ hóa YOLO (PyTorch) |
| `dich_yolov6.md` | YOLOv6 — Industrial Deployment (Re-param) |
| `dich_yolov7.md` | YOLOv7 — SOTA Accuracy (E-ELAN) |
| `dich_yolov8.md` | YOLOv8 — Đa năng nhất (Anchor-free, 6 tasks) |
| `dich_yolov9.md` | YOLOv9 — PGI & Information Bottleneck |
| `dich_yolov10.md` | YOLOv10 — NMS-Free Detection |
| `dich_yolo11.md` | YOLO11 — C3K2 & C2PSA |
| `dich_yolov12.md` | YOLOv12 — Attention-Centric |
| `dich_yoloe.md` | YOLOE — Open-Vocabulary Real-Time |
| `dich_yolo26.md` | YOLO26 — Edge-First Design (MuSGD, STAL) |

### 📊 Tài liệu tổng hợp
| File | Nội dung |
|:--|:--|
| `yolo_tong_hop.md` | Tổng hợp evolution từ v1 đến YOLO26 |
| `so_sanh_da_chieu.md` | So sánh đa chiều: mAP, FPS, Params |
| `yolo_bien_the.md` | Các biến thể: YOLOX, PP-YOLOE, YOLO-NAS, YOLO-World |

## 🏗️ Cấu trúc báo cáo LaTeX

```
baocao_yolo_v2/
├── main.tex              # File chính
├── main.pdf              # PDF output (48 trang)
├── chuong1_modau.tex     # Mở đầu
├── chuong2_nentang.tex   # Nền tảng lý thuyết
├── chuong3_yolov1v4.tex  # v1–v4
├── chuong4_yolov5v8.tex  # v5–v8
├── chuong5_yolov9v12.tex # v9–v12
├── chuong6_yolo26.tex    # YOLO26 (chương riêng)
├── chuong7_openvocab.tex # YOLOE + YOLO-World
├── chuong8_bienthe.tex   # Biến thể công nghiệp
├── chuong9_sosanhtonghop.tex # So sánh + Xu hướng
├── chuong10_ketluan.tex  # Kết luận + Tài liệu tham khảo
└── phuluc_thuatngu.tex   # Phụ lục thuật ngữ (40+ terms)
```

## ✨ Điểm nổi bật
- 🎯 **18 nguồn tài liệu tham khảo** khoa học
- 📐 **Công thức toán học** đầy đủ (Loss, MuSGD, STAL, ProgLoss)
- 🧠 **Ẩn dụ cho người không chuyên** (đóng khung tcolorbox)
- 📊 **Biểu đồ Pareto** (Params vs mAP)
- 🌳 **Sơ đồ phả hệ** YOLO Family Tree (Mermaid.js)
- ⚡ **Phân tích Edge AI** (MAC, NPU, INT8)

## 👤 Tác giả
**Nguyễn Cao Trị**

## 📜 License
Tài liệu này được tạo cho mục đích học tập và nghiên cứu.
