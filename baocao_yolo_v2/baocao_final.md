# KHáº¢O SÃT TOÃ€N DIá»†N Sá»° PHÃT TRIá»‚N Cá»¦A YOLO
## Tá»« YOLOv1 (2015) Ä‘áº¿n YOLO26 (2026): Kiáº¿n trÃºc â€” PhÆ°Æ¡ng phÃ¡p â€” Káº¿t quáº£ â€” Xu hÆ°á»›ng

**TÃ¡c giáº£:** Nguyá»…n Cao Trá»‹  
**NgÃ y biÃªn soáº¡n:** 14/05/2026

---

# Má»¤C Lá»¤C

1. [Má»Ÿ Ä‘áº§u](#chÆ°Æ¡ng-1-má»Ÿ-Ä‘áº§u)
2. [Ná»n táº£ng LÃ½ thuyáº¿t](#chÆ°Æ¡ng-2-ná»n-táº£ng-lÃ½-thuyáº¿t)
3. [Tháº¿ há»‡ Äáº§u: YOLOv1â€“v4 (2015â€“2020)](#chÆ°Æ¡ng-3-tháº¿-há»‡-Ä‘áº§u-yolov1v4)
4. [Tháº¿ há»‡ Giá»¯a: YOLOv5â€“v8 (2020â€“2023)](#chÆ°Æ¡ng-4-tháº¿-há»‡-giá»¯a-yolov5v8)
5. [Tháº¿ há»‡ Má»›i: YOLOv9â€“v12 (2024â€“2025)](#chÆ°Æ¡ng-5-tháº¿-há»‡-má»›i-yolov9v12)
6. [Má»Ÿ rá»™ng: YOLOE & YOLO26](#chÆ°Æ¡ng-6-má»Ÿ-rá»™ng-yoloe--yolo26)
7. [CÃ¡c biáº¿n thá»ƒ YOLO](#chÆ°Æ¡ng-7-cÃ¡c-biáº¿n-thá»ƒ-yolo)
8. [So sÃ¡nh Tá»•ng há»£p & ÄÃ¡nh giÃ¡](#chÆ°Æ¡ng-8-so-sÃ¡nh-tá»•ng-há»£p--Ä‘Ã¡nh-giÃ¡)
9. [Káº¿t luáº­n & Khuyáº¿n nghá»‹](#chÆ°Æ¡ng-9-káº¿t-luáº­n--khuyáº¿n-nghá»‹)
10. [Phá»¥ lá»¥c: Báº£ng Thuáº­t ngá»¯](#phá»¥-lá»¥c-báº£ng-thuáº­t-ngá»¯)
11. [TÃ i liá»‡u Tham kháº£o](#tÃ i-liá»‡u-tham-kháº£o)

---

# ChÆ°Æ¡ng 1: Má»ž Äáº¦U

## 1.1. Giá»›i thiá»‡u bÃ i toÃ¡n Object Detection

PhÃ¡t hiá»‡n váº­t thá»ƒ (Object Detection) lÃ  má»™t trong nhá»¯ng bÃ i toÃ¡n trung tÃ¢m cá»§a thá»‹ giÃ¡c mÃ¡y tÃ­nh (Computer Vision). Má»¥c tiÃªu lÃ  xÃ¡c Ä‘á»‹nh **vá»‹ trÃ­** (bounding box) vÃ  **loáº¡i** (class) cá»§a táº¥t cáº£ váº­t thá»ƒ trong áº£nh hoáº·c video. LÄ©nh vá»±c nÃ y Ä‘Æ°á»£c á»©ng dá»¥ng rá»™ng rÃ£i trong xe tá»± hÃ nh, giÃ¡m sÃ¡t an ninh, y táº¿, cÃ´ng nghiá»‡p vÃ  nÃ´ng nghiá»‡p.

TrÆ°á»›c khi YOLO ra Ä‘á»i (2015), cÃ¡c phÆ°Æ¡ng phÃ¡p detection chá»§ Ä‘áº¡o lÃ  **two-stage detectors**:

- **R-CNN** (2014): DÃ¹ng Selective Search táº¡o ~2000 vÃ¹ng Ä‘á» xuáº¥t (region proposals), má»—i vÃ¹ng qua CNN trÃ­ch features, rá»“i SVM phÃ¢n loáº¡i. Cá»±c cháº­m: **>40 giÃ¢y/áº£nh**.
- **Fast R-CNN** (2015): Cáº£i tiáº¿n báº±ng cÃ¡ch chia sáº» CNN features cho táº¥t cáº£ vÃ¹ng, nhÆ°ng váº«n phá»¥ thuá»™c Selective Search.
- **Faster R-CNN** (2015): Thay Selective Search báº±ng Region Proposal Network (RPN) há»c Ä‘Æ°á»£c, nhÆ°ng váº«n lÃ  pipeline **2 giai Ä‘oáº¡n rá»i ráº¡c**.

> **áº¨n dá»¥:** CÃ¡c phÆ°Æ¡ng phÃ¡p two-stage giá»‘ng nhÆ° **báº£o vá»‡ tÃ²a nhÃ  kiá»ƒm tra 2 láº§n**: láº§n 1 liáº¿c qua Ä‘Ã¡nh dáº¥u 2000 ngÆ°á»i "nghi ngá»", láº§n 2 gá»i tá»«ng ngÆ°á»i kiá»ƒm tra cÄƒn cÆ°á»›c â†’ chÃ­nh xÃ¡c nhÆ°ng cá»±c cháº­m.

## 1.2. YOLO â€” BÆ°á»›c ngoáº·t lá»‹ch sá»­

**YOLO (You Only Look Once)** ra Ä‘á»i nÄƒm 2015, thay Ä‘á»•i hoÃ n toÃ n cÃ¡ch tiáº¿p cáº­n: coi object detection lÃ  **bÃ i toÃ¡n há»“i quy duy nháº¥t** (single regression problem) â€” tá»« pixel áº£nh trá»±c tiáº¿p ra bounding boxes vÃ  class probabilities, chá»‰ qua **má»™t láº§n forward pass** â†’ real-time (45 FPS) [1].

> **áº¨n dá»¥:** YOLO giá»‘ng báº¡n **liáº¿c máº¯t 1 cÃ¡i** vÃ o phÃ²ng: "GÃ³c trÃ¡i cÃ³ cÃ¡i bÃ n, giá»¯a cÃ³ con mÃ¨o, bÃªn pháº£i cÃ³ cÃ¡i gháº¿" â€” biáº¿t ngay **cÃ¡i gÃ¬ + á»Ÿ Ä‘Ã¢u**, chá»‰ qua 1 láº§n nhÃ¬n.

Tá»« 2015 Ä‘áº¿n 2026, YOLO Ä‘Ã£ tráº£i qua **14 phiÃªn báº£n chÃ­nh** (v1â€“v12, YOLOE, YOLO26) vÃ  nhiá»u biáº¿n thá»ƒ, vá»›i mAP tÄƒng tá»« 63.4% (VOC) lÃªn ~57.2% (COCO â€” benchmark khÃ³ hÆ¡n nhiá»u), tá»‘c Ä‘á»™ tá»« 45 FPS lÃªn hÃ ng trÄƒm FPS.

## 1.3. Má»¥c tiÃªu vÃ  pháº¡m vi bÃ¡o cÃ¡o

**Má»¥c tiÃªu:** Kháº£o sÃ¡t toÃ n diá»‡n sá»± phÃ¡t triá»ƒn cá»§a YOLO qua cÃ¡c Ä‘á»i, phÃ¢n tÃ­ch kiáº¿n trÃºc, phÆ°Æ¡ng phÃ¡p, káº¿t quáº£, Æ°u/nhÆ°á»£c Ä‘iá»ƒm, vÃ  xu hÆ°á»›ng tÆ°Æ¡ng lai.

**Pháº¡m vi:**
- DÃ²ng chÃ­nh: YOLOv1 â†’ YOLOv12, YOLOE, YOLO26
- Biáº¿n thá»ƒ: YOLOX, PP-YOLOE, YOLO-NAS, YOLO-World
- So sÃ¡nh Ä‘a chiá»u vÃ  khuyáº¿n nghá»‹ lá»±a chá»n

---

# ChÆ°Æ¡ng 2: Ná»€N Táº¢NG LÃ THUYáº¾T

## 2.1. Pipeline Object Detection hiá»‡n Ä‘áº¡i

Má»™t detector hiá»‡n Ä‘áº¡i gá»“m 3 pháº§n:

| ThÃ nh pháº§n | Chá»©c nÄƒng | VÃ­ dá»¥ |
|:--|:--|:--|
| **Backbone** | TrÃ­ch xuáº¥t features tá»« áº£nh | Darknet, CSPDarknet, ResNet |
| **Neck** | Káº¿t há»£p features á»Ÿ nhiá»u scales | FPN, PAN, BiFPN |
| **Head** | Dá»± Ä‘oÃ¡n bounding boxes vÃ  classes | Coupled / Decoupled head |

> **áº¨n dá»¥:** Backbone giá»‘ng **máº¯t** (nhÃ¬n vÃ  nháº­n diá»‡n Ä‘áº·c trÆ°ng), Neck giá»‘ng **nÃ£o bá»™** (tá»•ng há»£p thÃ´ng tin tá»« nhiá»u gÃ³c nhÃ¬n), Head giá»‘ng **miá»‡ng** (nÃ³i ra Ä‘Ã¡p Ã¡n cuá»‘i cÃ¹ng).

[ðŸ“Œ YÃŠU Cáº¦U CHÃˆN áº¢NH: TÃ¡c giáº£ hÃ£y chÃ¨n sÆ¡ Ä‘á»“ khá»‘i tá»•ng quÃ¡t Backbone â†’ Neck â†’ Head cá»§a má»™t YOLO detector hiá»‡n Ä‘áº¡i.]

## 2.2. One-stage vs Two-stage Detectors

| TiÃªu chÃ­ | Two-stage (Faster R-CNN) | One-stage (YOLO, SSD) |
|:--|:--|:--|
| Tá»‘c Ä‘á»™ | Cháº­m (5â€“20 FPS) | Nhanh (30â€“300 FPS) |
| Accuracy | Cao hÆ¡n (trÆ°á»›c 2020) | Báº¯t ká»‹p tá»« YOLOv4+ [4] |
| Pipeline | 2 giai Ä‘oáº¡n rá»i ráº¡c | End-to-end |

## 2.3. CÃ¡c khÃ¡i niá»‡m cá»‘t lÃµi

### 2.3.1. IoU (Intersection over Union)

Äo má»©c trÃ¹ng láº·p giá»¯a predicted box (B_p) vÃ  ground truth (B_gt):

$$\text{IoU} = \frac{|B_p \cap B_{gt}|}{|B_p \cup B_{gt}|}$$

> **áº¨n dá»¥:** TÆ°á»Ÿng tÆ°á»£ng 2 tá» giáº¥y Ä‘áº·t chá»“ng lÃªn nhau. IoU = *diá»‡n tÃ­ch pháº§n chá»“ng* chia cho *tá»•ng diá»‡n tÃ­ch cáº£ 2 tá» (Ä‘áº¿m 1 láº§n)*. IoU = 1.0 nghÄ©a lÃ  2 tá» trÃ¹ng khÃ­t hoÃ n háº£o.

### 2.3.2. mAP (mean Average Precision)

- **mAP@50:** TÃ­nh Average Precision á»Ÿ ngÆ°á»¡ng IoU â‰¥ 0.5. Metric chÃ­nh trÃªn PASCAL VOC.
- **mAP@50:95:** Trung bÃ¬nh AP á»Ÿ IoU tá»« 0.5 Ä‘áº¿n 0.95 (bÆ°á»›c 0.05) â€” metric chÃ­nh trÃªn MS COCO, kháº¯t khe hÆ¡n nhiá»u.

### 2.3.3. Anchor Boxes

CÃ¡c há»™p máº«u (templates) Ä‘Æ°á»£c Ä‘á»‹nh nghÄ©a trÆ°á»›c vá»›i kÃ­ch thÆ°á»›c vÃ  tá»· lá»‡ khÃ¡c nhau. Model dá»± Ä‘oÃ¡n **offsets** so vá»›i anchors thay vÃ¬ tá»a Ä‘á»™ tuyá»‡t Ä‘á»‘i.

> **áº¨n dá»¥:** Anchor giá»‘ng Ä‘i **may Ä‘á»“ chá»n size S/M/L cÃ³ sáºµn**, rá»“i chá»‰ cáº§n sá»­a lai cho vá»«a â€” nhanh vÃ  chÃ­nh xÃ¡c hÆ¡n so vá»›i Ä‘o tá»« Ä‘áº§u (anchor-free).

### 2.3.4. NMS (Non-Maximum Suppression)

Thuáº­t toÃ¡n loáº¡i bá» bounding boxes trÃ¹ng láº·p: giá»¯ box cÃ³ confidence cao nháº¥t, bá» cÃ¡c boxes cÃ³ IoU lá»›n vá»›i box Ä‘Ã£ chá»n.

### 2.3.5. Tiáº¿n hÃ³a Loss Function

| Loss | PhiÃªn báº£n | Äáº·c Ä‘iá»ƒm |
|:--|:--|:--|
| MSE/SSE | v1â€“v3 | ÄÆ¡n giáº£n, coi box lá»›n/nhá» ngang nhau |
| GIoU | v6 | XÃ©t cáº£ vÃ¹ng bao ngoÃ i khi 2 box khÃ´ng giao |
| CIoU | v4â€“v8 | + khoáº£ng cÃ¡ch tÃ¢m + tá»· lá»‡ khung hÃ¬nh |
| DFL | v8â€“v12 | Dá»± Ä‘oÃ¡n phÃ¢n phá»‘i xÃ¡c suáº¥t thay vÃ¬ giÃ¡ trá»‹ Ä‘Æ¡n |
| Bá» DFL | YOLO26 | Quay vá» dá»± Ä‘oÃ¡n trá»±c tiáº¿p, bÃ¹ báº±ng training tricks |

### 2.3.6. Datasets chuáº©n

- **PASCAL VOC** (2007/2012): 20 classes, ~10K áº£nh. Metric: mAP@50.
- **MS COCO:** 80 classes, 330K áº£nh. Metric: mAP@50:95. Benchmark chÃ­nh tá»« v3+.
- **LVIS:** 1203 categories. DÃ¹ng cho open-vocabulary evaluation (YOLOE) [13].

---

# ChÆ°Æ¡ng 3: THáº¾ Há»† Äáº¦U â€” YOLOv1â€“v4 (2015â€“2020)

## 3.1. YOLOv1 (2015) â€” Khá»Ÿi nguá»“n

**Paper:** "You Only Look Once: Unified, Real-Time Object Detection" (CVPR 2016) [1]  
**TÃ¡c giáº£:** Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi

### 3.1.1. Ã tÆ°á»Ÿng cá»‘t lÃµi

Chia áº£nh thÃ nh lÆ°á»›i **S Ã— S** (S=7), má»—i Ã´ dá»± Ä‘oÃ¡n **B=2** bounding boxes vÃ  **C=20** class probabilities. ToÃ n bá»™ pipeline lÃ  **má»™t máº¡ng neural duy nháº¥t**, end-to-end.

### 3.1.2. Kiáº¿n trÃºc

24 lá»›p convolutional + 2 lá»›p fully connected. Output: tensor **7 Ã— 7 Ã— 30**.

[ðŸ“Œ YÃŠU Cáº¦U CHÃˆN áº¢NH: TÃ¡c giáº£ hÃ£y chÃ¨n sÆ¡ Ä‘á»“ kiáº¿n trÃºc YOLOv1 tá»« paper gá»‘c (Figure 3, Redmon et al. 2016) [1].]

### 3.1.3. HÃ m Loss (5 thÃ nh pháº§n)

$$\mathcal{L} = \lambda_{\text{coord}} \sum_{i=0}^{S^2}\sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right]$$

$$+ \sum_{i}\sum_{j} \mathbb{1}_{ij}^{\text{obj}}(C_i - \hat{C}_i)^2 + \lambda_{\text{noobj}} \sum_{i}\sum_{j} \mathbb{1}_{ij}^{\text{noobj}}(C_i - \hat{C}_i)^2 + \sum_{i} \mathbb{1}_{i}^{\text{obj}} \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2$$

Trong Ä‘Ã³: Î»_coord = 5, Î»_noobj = 0.5. DÃ¹ng âˆšw, âˆšh Ä‘á»ƒ **giáº£m áº£nh hÆ°á»Ÿng sai sá»‘** á»Ÿ box lá»›n [1].

### 3.1.4. Káº¿t quáº£

| Model | mAP (VOC 2007) | FPS |
|:--|:-:|:-:|
| Fast YOLO | 52.7% | 155 |
| YOLO | 63.4% | 45 |
| YOLO VGG-16 | 66.4% | 21 |
| Faster R-CNN | 70.0% | 0.5 |

*Ghi chÃº: FPS Ä‘o trÃªn Titan X (Maxwell). KhÃ´ng so sÃ¡nh trá»±c tiáº¿p vá»›i FPS tháº¿ há»‡ GPU sau.*

**Æ¯u Ä‘iá»ƒm:** Real-time Ä‘áº§u tiÃªn; Ã­t false positive background; tá»•ng quÃ¡t hÃ³a tá»‘t (Picasso dataset: 53.3% vs R-CNN 10.2%) [1].  
**NhÆ°á»£c Ä‘iá»ƒm:** Váº­t nhá» kÃ©m (grid thÃ´ 7Ã—7); localization sai nhiá»u; má»—i Ã´ chá»‰ dá»± Ä‘oÃ¡n 1 class.

---

## 3.2. YOLOv2 / YOLO9000 (2016) â€” Better, Faster, Stronger

**Paper:** "YOLO9000: Better, Faster, Stronger" (CVPR 2017) [2]  
**TÃ¡c giáº£:** Joseph Redmon, Ali Farhadi

### 3.2.1. Cáº£i tiáº¿n tÃ­ch lÅ©y

| # | Ká»¹ thuáº­t | mAP (%) | Thay Ä‘á»•i |
|:-:|:--|:-:|:--|
| 0 | Baseline (v1) | 63.4 | â€” |
| 1 | + Batch Normalization | 65.4 | +2.0 |
| 2 | + High-Res Classifier (448) | 69.4 | +4.0 |
| 3 | + Anchor Boxes (5 anchors) | 69.2 | Recall +7% |
| 4 | + Dimension Clusters (k-means) | â€” | IoU tá»‘t hÆ¡n |
| 5 | + Direct Location (sigmoid) | 74.4 | +5.0 |
| 6 | + Passthrough (26Ã—26â†’13Ã—13) | 75.4 | +1.0 |
| 7 | + Multi-Scale Training | 76.8 | +1.4 |
| | **YOLOv2 (416Ã—416)** | **76.8** | **+13.4** |

### 3.2.2. Darknet-19 & YOLO9000

Backbone má»›i: 19 conv + 5 maxpool, chá»‰ **5.58B FLOPs** (VGG-16: 30.69B â€” nhanh gáº¥p 5Ã—). YOLO9000 dÃ¹ng **WordTree** joint training trÃªn COCO + ImageNet â†’ detect **9000+ categories** [2].

---

## 3.3. YOLOv3 (2018) â€” Multi-Scale Detection

**Paper:** "YOLOv3: An Incremental Improvement" (Tech Report) [3]  
**TÃ¡c giáº£:** Joseph Redmon, Ali Farhadi *(paper cuá»‘i cÃ¹ng cá»§a Redmon trÆ°á»›c khi dá»«ng nghiÃªn cá»©u CV vÃ¬ lo ngáº¡i Ä‘áº¡o Ä‘á»©c)*

### 3.3.1. Äá»™t phÃ¡: 3 Scales

Dá»± Ä‘oÃ¡n á»Ÿ **3 scales** (13Ã—13, 26Ã—26, 52Ã—52) theo FPN-style, má»—i scale 3 anchors = 9 anchors tá»•ng. Thay Softmax báº±ng **Sigmoid** cho multi-label classification [3].

> **áº¨n dá»¥:** v1â€“v2 nhÃ¬n qua 1 á»‘ng kÃ­nh. v3 nhÃ¬n qua **3 á»‘ng kÃ­nh zoom khÃ¡c nhau** cÃ¹ng lÃºc: zoom xa (váº­t lá»›n), trung bÃ¬nh, zoom gáº§n (váº­t nhá»).

### 3.3.2. Darknet-53

53 conv layers + residual connections. ChÃ­nh xÃ¡c báº±ng ResNet-152 nhÆ°ng **nhanh gáº¥p 2Ã—** (78 vs 37 FPS) [3].

**Káº¿t quáº£:** mAP@50: 57.9% (COCO), ngang RetinaNet nhÆ°ng nhanh gáº¥p 3.8Ã—. mAP@50:95: 33.0%.

---

## 3.4. YOLOv4 (2020) â€” Táº­p Ä‘áº¡i thÃ nh

**Paper:** "YOLOv4: Optimal Speed and Accuracy of Object Detection" [4]  
**TÃ¡c giáº£:** Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao

### 3.4.1. PhÆ°Æ¡ng phÃ¡p luáº­n: BoF & BoS

| NhÃ³m | Ã nghÄ©a | Ká»¹ thuáº­t tiÃªu biá»ƒu |
|:--|:--|:--|
| **Bag of Freebies** | Chá»‰ tÄƒng training cost | Mosaic, CutMix, CIoU Loss, Label Smoothing, SAT |
| **Bag of Specials** | TÄƒng nháº¹ inference cost | Mish activation, SPP, SAM, PAN, DIoU-NMS |

> **áº¨n dá»¥:** BoF giá»‘ng **Ä‘áº§u báº¿p gom táº¥t cáº£ cÃ´ng thá»©c hay nháº¥t tháº¿ giá»›i** vÃ o 1 cuá»‘n sÃ¡ch. KhÃ´ng cáº§n phÃ¡t minh má»›i, chá»‰ cáº§n biáº¿t chá»n vÃ  káº¿t há»£p Ä‘Ãºng.

### 3.4.2. Kiáº¿n trÃºc: CSPDarknet53 + SPP + PAN

- **CSP (Cross Stage Partial):** Chia features thÃ nh 2 pháº§n, chá»‰ 1 pháº§n qua processing â†’ giáº£m **20% computation** mÃ  giá»¯ accuracy.
- **SPP:** MaxPool nhiá»u kÃ­ch thÆ°á»›c â†’ tÄƒng receptive field.
- **PAN:** Bottom-up pathway bá»• sung cho FPN â†’ localization signals máº¡nh hÆ¡n.

[ðŸ“Œ YÃŠU Cáº¦U CHÃˆN áº¢NH: TÃ¡c giáº£ hÃ£y chÃ¨n sÆ¡ Ä‘á»“ kiáº¿n trÃºc YOLOv4 (CSPDarknet53 + SPP + PAN) tá»« paper gá»‘c [4].]

### 3.4.3. Ká»¹ thuáº­t ná»•i báº­t

**Mosaic Augmentation:** GhÃ©p 4 áº£nh ngáº«u nhiÃªn thÃ nh 1 áº£nh â†’ model tháº¥y nhiá»u context, giáº£m batch size cáº§n thiáº¿t [4].

**CIoU Loss:**

$$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2} + \alpha v$$

Trong Ä‘Ã³: Ï = khoáº£ng cÃ¡ch Euclid giá»¯a 2 tÃ¢m, c = Ä‘Æ°á»ng chÃ©o vÃ¹ng bao, v = há»‡ sá»‘ aspect ratio, Î± = há»‡ sá»‘ cÃ¢n báº±ng [4].

> **áº¨n dá»¥:** IoU chá»‰ Ä‘o "2 box chá»“ng bao nhiÃªu %". CIoU Ä‘o thÃªm "2 tÃ¢m box cÃ¡ch nhau bao xa" vÃ  "hÃ¬nh dáº¡ng 2 box cÃ³ giá»‘ng nhau khÃ´ng" â†’ toÃ n diá»‡n hÆ¡n.

### 3.4.4. Káº¿t quáº£

**43.5% mAP** (COCO), ~65 FPS (V100). Nháº£y vá»t **+10.5%** so vá»›i v3 â€” bÆ°á»›c nháº£y lá»›n nháº¥t lá»‹ch sá»­ YOLO [4]. Train Ä‘Æ°á»£c trÃªn **single GPU** (8â€“16 GB).

*Ghi chÃº: FPS Ä‘o trÃªn V100 GPU. KhÃ´ng so sÃ¡nh trá»±c tiáº¿p vá»›i FPS Ä‘o trÃªn Titan X (v1â€“v3).*
# ChÆ°Æ¡ng 4: THáº¾ Há»† GIá»®A â€” YOLOv5â€“v8 (2020â€“2023)

## 4.1. YOLOv5 (2020) â€” DÃ¢n chá»§ hÃ³a YOLO

**TÃ¡c giáº£:** Glenn Jocher / Ultralytics | **Framework:** PyTorch | *KhÃ´ng cÃ³ paper chÃ­nh thá»©c* [5]

### 4.1.1. Ã nghÄ©a lá»‹ch sá»­

Chuyá»ƒn tá»« Darknet (C) sang PyTorch. GÃ¢y tranh cÃ£i vÃ¬ khÃ´ng cÃ³ paper, nhÆ°ng trá»Ÿ thÃ nh **YOLO phá»• biáº¿n nháº¥t** nhá» `pip install ultralytics`.

> **áº¨n dá»¥:** TrÆ°á»›c v5, dÃ¹ng YOLO giá»‘ng pháº£i **lÃ¡i xe sá»‘ sÃ n** (compile C code, cáº¥u hÃ¬nh .cfg). v5 chuyá»ƒn sang **xe sá»‘ tá»± Ä‘á»™ng** â€” 3 dÃ²ng Python lÃ  cháº¡y.

### 4.1.2. Kiáº¿n trÃºc

- **Backbone:** CSPDarknet53 (modified) vá»›i **C3 module** â€” CSP Bottleneck 3 convolutions
- **Neck:** PANet + **SPPF** (SPP Fast) â€” 3 MaxPool $5 \times 5$ ná»‘i tiáº¿p thay vÃ¬ 3 MaxPool song song â†’ nhanh hÆ¡n ~2Ã— trÃªn GPU
- **Head:** Anchor-based, 3 scales, **AutoAnchor** (k-means + genetic algorithm tá»± tÃ¬m anchors tá»‘i Æ°u)

> **áº¨n dá»¥ SPPF:** SPP giá»‘ng 3 ngÆ°á»i **cÃ¹ng lÃºc** Ä‘á»c 3 cuá»‘n sÃ¡ch â†’ cáº§n 3 bÃ n. SPPF giá»‘ng 1 ngÆ°á»i Ä‘á»c **ná»‘i tiáº¿p** 3 cuá»‘n â†’ chá»‰ cáº§n 1 bÃ n, káº¿t quáº£ tÆ°Æ¡ng Ä‘Æ°Æ¡ng.

### 4.1.3. Káº¿t quáº£

| Model | Params | FLOPs | mAP@50:95 | PhÃ¹ há»£p |
|:--|:-:|:-:|:-:|:--|
| v5n | 1.9M | 4.5G | 28.0% | Mobile, IoT |
| v5s | 7.2M | 16.5G | 37.4% | Edge devices |
| v5m | 21.2M | 49.0G | 45.4% | CÃ¢n báº±ng |
| v5l | 46.5M | 109.1G | 49.0% | Server |
| v5x | 86.7M | 205.7G | 50.7% | Max accuracy |

*Ghi chÃº: Äo trÃªn COCO val2017, input 640Ã—640 [5].*

---

## 4.2. YOLOv6 (2022) â€” Industrial Deployment

**Paper:** "YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications" [6]  
**TÃ¡c giáº£:** Meituan Vision AI

### 4.2.1. Re-parameterization â€” Ká»¹ thuáº­t chá»§ Ä‘áº¡o

Kiáº¿n trÃºc **khÃ¡c nhau giá»¯a training vÃ  inference**:

- **Training:** Multi-branch â€” Conv $3 \times 3$ + Conv $1 \times 1$ + Identity song song â†’ há»c phong phÃº
- **Inference:** Gá»™p (re-parameterize) thÃ nh **1 Conv $3 \times 3$ duy nháº¥t** â†’ nhanh
- **ToÃ¡n:** $W_{\text{merged}} = W_{3 \times 3} + \text{pad}(W_{1 \times 1}) + I$

> **áº¨n dá»¥:** Giá»‘ng **Ã´n thi** â€” khi Ã´n, báº¡n Ä‘á»c 5 cuá»‘n sÃ¡ch, ghi chÃº, lÃ m bÃ i táº­p (multi-branch). Khi vÃ o phÃ²ng thi, báº¡n chá»‰ mang **1 tá» tÃ³m táº¯t** chá»©a tinh hoa tá»« 5 cuá»‘n (single branch).

[ðŸ“Œ YÃŠU Cáº¦U CHÃˆN áº¢NH: TÃ¡c giáº£ hÃ£y chÃ¨n sÆ¡ Ä‘á»“ Re-parameterization (training multi-branch â†’ inference single-branch) tá»« paper YOLOv6 [6].]

### 4.2.2. Kiáº¿n trÃºc chi tiáº¿t

| ThÃ nh pháº§n | Ká»¹ thuáº­t | Giáº£i thÃ­ch |
|:--|:--|:--|
| Backbone | EfficientRep | RepVGG blocks, re-parameterizable |
| Neck | Rep-PAN | PAN + RepVGG blocks |
| Head | Efficient Decoupled | TÃ¡ch cls/reg (giá»‘ng YOLOX), giáº£m 1 Conv |
| Assignment | SimOTA â†’ TAL | Task-Aligned Learning |
| Loss | VFL + SIoU/GIoU + DFL | Varifocal Loss cho classification |
| Anchor | **Anchor-free** | FCOS-style, 4 distances tá»« tÃ¢m |

### 4.2.3. Quantization Pipeline â€” Äiá»ƒm máº¡nh nháº¥t

INT8 chá»‰ máº¥t **0.2% mAP** nhÆ°ng tÄƒng **75% FPS** [6]. DÃ¹ng PTQ + QAT + RepOptimizer.

### 4.2.4. Káº¿t quáº£

| Model | mAP@50:95 | FPS (T4 TRT FP16) | Params | So vá»›i v5 |
|:--|:-:|:-:|:-:|:--|
| YOLOv5-N | 28.0% | 602 | 1.9M | â€” |
| **YOLOv6-N** | **37.5%** | **1187** | **4.7M** | +9.5%, nhanh 2Ã— |
| YOLOv5-S | 37.4% | 349 | 7.2M | â€” |
| **YOLOv6-S** | **45.0%** | **495** | **18.5M** | +7.6% |
| YOLOv5-L | 49.0% | 113 | 46.5M | â€” |
| **YOLOv6-L** | **52.8%** | **116** | **59.6M** | +3.8% |

*Ghi chÃº: FPS Ä‘o trÃªn T4 GPU, TensorRT FP16. KhÃ´ng so sÃ¡nh trá»±c tiáº¿p vá»›i FPS Ä‘o trÃªn V100 hoáº·c Titan X á»Ÿ tháº¿ há»‡ trÆ°á»›c [6].*

---

## 4.3. YOLOv7 (2022) â€” SOTA Accuracy

**Paper:** "YOLOv7: Trainable Bag-of-Freebies Sets New SOTA for Real-Time Object Detectors" (CVPR 2023) [7]  
**TÃ¡c giáº£:** Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao

### 4.3.1. E-ELAN (Extended Efficient Layer Aggregation Network)

Má»Ÿ rá»™ng ELAN báº±ng cÃ¡ch tÄƒng **cardinality** (sá»‘ nhÃ¡nh song song) mÃ  **khÃ´ng phÃ¡ gradient path gá»‘c** [7].

> **áº¨n dá»¥:** ELAN giá»‘ng 1 dÃ¢y chuyá»n sáº£n xuáº¥t vá»›i 3 cÃ´ng nhÃ¢n ná»‘i tiáº¿p. E-ELAN thÃªm "ca kÃ­p" má»›i (expand cardinality) nhÆ°ng giá»¯ nguyÃªn dÃ¢y chuyá»n cÅ©, rá»“i **shuffle + merge** káº¿t quáº£ â†’ output phong phÃº hÆ¡n.

### 4.3.2. Planned Re-parameterized Conv

RepConv (RepVGG) **khÃ´ng nÃªn dÃ¹ng** khi cÃ³ identity connection â†’ xung Ä‘á»™t gradient. **RepConvN:** Bá» identity branch, chá»‰ giá»¯ conv $1 \times 1$ + conv $3 \times 3$. TÄƒng 0.5% AP [7].

### 4.3.3. Coarse-to-Fine Label Assignment

- **Lead Head** (fine-grained): GÃ¡n nhÃ£n cháº·t cháº½ â€” chÃ­nh xÃ¡c cao
- **Auxiliary Head** (coarse-grained): GÃ¡n nhÃ£n lá»ng hÆ¡n â†’ nhiá»u positive samples â†’ train tá»‘t hÆ¡n. **Chá»‰ dÃ¹ng khi training, bá» khi inference** â†’ zero cost [7]

> **áº¨n dá»¥:** Lead head giá»‘ng **giÃ¡o viÃªn cháº¥m thi nghiÃªm kháº¯c** (chá»‰ cho Ä‘iá»ƒm khi Ä‘Ãºng hoÃ n toÃ n). Auxiliary head giá»‘ng trá»£ giáº£ng nháº¹ nhÃ ng (cho Ä‘iá»ƒm khi gáº§n Ä‘Ãºng) â†’ sinh viÃªn nháº­n nhiá»u feedback â†’ há»c tá»‘t hÆ¡n.

### 4.3.4. Káº¿t quáº£

**Base models (V100 GPU):**

| Model | Params | FPS | mAP@50:95 | So sÃ¡nh |
|:--|:-:|:-:|:-:|:--|
| YOLOv5-L | 46.5M | 99 | 48.8% | â€” |
| YOLOX-L | 54.2M | 68 | 50.0% | â€” |
| PPYOLOE-L | 52.2M | 78 | 51.4% | â€” |
| **YOLOv7** | **36.9M** | **161** | **51.4%** | Ãt params nháº¥t, nhanh nháº¥t |
| **YOLOv7-X** | **71.3M** | **114** | **53.1%** | VÆ°á»£t PPYOLOE |

**Large models â€” SOTA:**

| Model | FPS | mAP@50:95 |
|:--|:-:|:-:|
| YOLOv7-W6 | 84 | 54.9% |
| YOLOv7-E6 | 56 | 55.9% |
| **YOLOv7-E6E** | **36** | **56.8%** |
| SWIN-L Cascade R-CNN | 12 | 53.9% |

*Ghi chÃº: FPS Ä‘o trÃªn V100 GPU. v7-E6E (56.8%) vÆ°á»£t táº¥t cáº£ detectors táº¡i thá»i Ä‘iá»ƒm Ä‘Ã³, ká»ƒ cáº£ Transformer-based [7].*

---

## 4.4. YOLOv8 (2023) â€” Äa nÄƒng nháº¥t

**TÃ¡c giáº£:** Ultralytics | *KhÃ´ng cÃ³ paper chÃ­nh thá»©c* [8]

### 4.4.1. Ba thay Ä‘á»•i kiáº¿n trÃºc lá»›n so vá»›i v5

**1. Anchor-Free:** Bá» hoÃ n toÃ n anchor boxes. Dá»± Ä‘oÃ¡n **center point + 4 distances** (khoáº£ng cÃ¡ch tá»« tÃ¢m Ä‘áº¿n 4 cáº¡nh).

> **áº¨n dá»¥:** Anchor-based giá»‘ng chá»n **khung áº£nh S/M/L** rá»“i chá»‰nh. Anchor-free giá»‘ng **váº½ tá»± do** â€” Ä‘á»©ng táº¡i tÃ¢m váº­t, Ä‘o 4 hÆ°á»›ng lÃªn/xuá»‘ng/trÃ¡i/pháº£i.

**2. Decoupled Head:** TÃ¡ch classification vÃ  regression thÃ nh 2 branches riÃªng biá»‡t [8].

**3. DFL (Distribution Focal Loss):** Thay vÃ¬ dá»± Ä‘oÃ¡n 1 giÃ¡ trá»‹ cho má»—i tá»a Ä‘á»™ box, dá»± Ä‘oÃ¡n **phÃ¢n phá»‘i xÃ¡c suáº¥t** trÃªn 16 bins.

> **áº¨n dá»¥:** Thay vÃ¬ nÃ³i "cáº¡nh trÃ¡i cÃ¡ch tÃ¢m 25px" (1 sá»‘ duy nháº¥t), DFL nÃ³i "xÃ¡c suáº¥t 10% á»Ÿ 24px, 70% á»Ÿ 25px, 20% á»Ÿ 26px" â†’ pháº£n Ã¡nh **má»©c Ä‘á»™ khÃ´ng cháº¯c cháº¯n**, localization chÃ­nh xÃ¡c hÆ¡n.

### 4.4.2. C2f Module

Káº¿ thá»«a C3 (v5) + cáº£m há»©ng ELAN (v7): **concat output cá»§a Táº¤T Cáº¢ bottlenecks** thay vÃ¬ chá»‰ output cuá»‘i â†’ gradient flow phong phÃº hÆ¡n.

[ðŸ“Œ YÃŠU Cáº¦U CHÃˆN áº¢NH: TÃ¡c giáº£ hÃ£y chÃ¨n sÆ¡ Ä‘á»“ so sÃ¡nh C3 (v5) vs C2f (v8), thá»ƒ hiá»‡n gradient flow vÃ  feature concatenation.]

### 4.4.3. Multi-task â€” 6 tÃ¡c vá»¥

| TÃ¡c vá»¥ | Output | VÃ­ dá»¥ á»©ng dá»¥ng |
|:--|:--|:--|
| Detection | Bounding boxes | PhÃ¡t hiá»‡n xe, ngÆ°á»i |
| Segmentation | Pixel-level masks | TÃ¡ch ná»n áº£nh |
| Classification | Class label | PhÃ¢n loáº¡i sáº£n pháº©m |
| Pose Estimation | 17 Keypoints | PhÃ¢n tÃ­ch tÆ° tháº¿ |
| OBB | Rotated boxes | áº¢nh vá»‡ tinh |
| Tracking | Box + Track ID | Äáº¿m ngÆ°á»i qua cá»­a |

### 4.4.4. Káº¿t quáº£

| Model | Params | mAP@50:95 | So vá»›i v5 |
|:--|:-:|:-:|:--|
| v8n | 3.2M | 37.3% | v5n: 28.0% (**+9.3%**) |
| v8s | 11.2M | 44.9% | v5s: 37.4% (**+7.5%**) |
| v8m | 25.9M | 50.2% | v5m: 45.4% (**+4.8%**) |
| v8l | 43.7M | 52.9% | v5l: 49.0% (**+3.9%**) |
| v8x | 68.2M | 53.9% | v5x: 50.7% (**+3.2%**) |

âš ï¸ **LÆ°u Ã½ quan trá»ng:** v8-X (53.9%) **thua** v7-E6E (56.8%) vá» mAP thuáº§n tÃºy. Tuy nhiÃªn v8 tháº¯ng á»Ÿ: ecosystem (â˜…â˜…â˜…â˜…â˜…), multi-task (6 tÃ¡c vá»¥), API thá»‘ng nháº¥t [7][8].

**â†’ Náº¿u cáº§n accuracy tá»‘i Ä‘a â†’ v7. Náº¿u cáº§n Ä‘a nÄƒng + dá»… dÃ¹ng â†’ v8.**
# ChÆ°Æ¡ng 5: THáº¾ Há»† Má»šI â€” YOLOv9â€“v12 (2024â€“2025)

## 5.1. YOLOv9 (2024) â€” Chá»‘ng máº¥t thÃ´ng tin

**Paper:** "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information" [9]  
**TÃ¡c giáº£:** Chien-Yao Wang, I-Hau Yeh, Hong-Yuan Mark Liao

### 5.1.1. Váº¥n Ä‘á»: Information Bottleneck

Khi data Ä‘i qua nhiá»u layers, **mutual information giáº£m Ä‘Æ¡n Ä‘iá»‡u**:

$$I(X, X) \geq I(X, f_1(X)) \geq I(X, f_2(f_1(X))) \geq \ldots$$

> **áº¨n dá»¥:** Giá»‘ng trÃ² **truyá»n tin** â€” ngÆ°á»i Ä‘áº§u nháº­n tin gá»‘c, truyá»n miá»‡ng qua 20 ngÆ°á»i, ngÆ°á»i cuá»‘i nháº­n tin sai lá»‡ch. Layers sÃ¢u "quÃªn" thÃ´ng tin input gá»‘c â†’ gradient khÃ´ng Ä‘Ã¡ng tin cáº­y [9].

### 5.1.2. PGI (Programmable Gradient Information)

3 thÃ nh pháº§n:

1. **Main Branch:** Network chÃ­nh, dÃ¹ng cho inference. Kiáº¿n trÃºc tÃ¹y chá»n.
2. **Auxiliary Reversible Branch:** Máº¡ng phá»¥ cÃ³ kháº£ nÄƒng **khÃ´i phá»¥c input tá»« output** â†’ giá»¯ complete information. **Chá»‰ dÃ¹ng khi training, bá» khi inference** â†’ zero cost.
3. **Multi-level Auxiliary:** Inject gradient á»Ÿ **nhiá»u táº§ng** â€” trÃ¡nh gradient bá»‹ "Ä‘á»©t".

> **áº¨n dá»¥:** Main branch giá»‘ng "truyá»n miá»‡ng" qua 20 ngÆ°á»i (máº¥t thÃ´ng tin). PGI thÃªm 1 **kÃªnh SMS song song** (auxiliary branch) â€” má»—i ngÆ°á»i Ä‘á»u nháº­n SMS Ä‘á»‘i chiáº¿u â†’ sá»­a sai ká»‹p thá»i. SMS **chá»‰ dÃ¹ng lÃºc luyá»‡n táº­p**, khi "lÃªn sÃ¢n kháº¥u" (inference) thÃ¬ bá».

[ðŸ“Œ YÃŠU Cáº¦U CHÃˆN áº¢NH: TÃ¡c giáº£ hÃ£y chÃ¨n sÆ¡ Ä‘á»“ PGI (Main Branch + Auxiliary Reversible Branch) tá»« paper YOLOv9 (Figure 4) [9].]

### 5.1.3. GELAN (Generalized ELAN)

Tá»•ng quÃ¡t hÃ³a CSPNet + ELAN â€” cho phÃ©p dÃ¹ng **báº¥t ká»³ computational block** nÃ o (Conv, Bottleneck, Transformer...) [9].

> **áº¨n dá»¥:** ELAN cÅ© giá»‘ng dÃ¢y chuyá»n chá»‰ láº¯p Ä‘Æ°á»£c **1 loáº¡i mÃ¡y**. GELAN giá»‘ng dÃ¢y chuyá»n "universal" â€” láº¯p mÃ¡y gÃ¬ vÃ o cÅ©ng cháº¡y Ä‘Æ°á»£c.

### 5.1.4. Káº¿t quáº£

| Model | Params | FLOPs | mAP@50:95 | So vá»›i v8 |
|:--|:-:|:-:|:-:|:--|
| v8-S | 11.2M | 28.6G | 44.9% | â€” |
| **v9-S** | **7.2M** | **26.7G** | **46.8%** | +1.9%, Ã­t params hÆ¡n |
| v8-M | 25.9M | 78.9G | 50.2% | â€” |
| **v9-C** | **25.5M** | **102.8G** | **53.0%** | +2.8% |
| v8-X | 68.2M | 257.8G | 53.9% | â€” |
| **v9-E** | **58.1M** | **192.5G** | **55.6%** | +1.7%, Ã­t params hÆ¡n |

*Ghi chÃº: Äo trÃªn COCO val2017. v9 luÃ´n Ã­t params hÆ¡n v8 mÃ  accuracy cao hÆ¡n á»Ÿ má»i scale â€” chá»©ng minh PGI hiá»‡u quáº£ [9].*

---

## 5.2. YOLOv10 (2024) â€” NMS-Free

**Paper:** "YOLOv10: Real-Time End-to-End Object Detection" [10]  
**TÃ¡c giáº£:** Ao Wang et al. (Tsinghua University)

### 5.2.1. Váº¥n Ä‘á» NMS

NMS cÃ³ 3 nhÆ°á»£c Ä‘iá»ƒm: (1) Cáº§n tune threshold, (2) KhÃ´ng differentiable, (3) Latency phá»¥ thuá»™c data â€” áº£nh nhiá»u váº­t â†’ NMS cháº­m hÆ¡n â†’ **latency khÃ´ng á»•n Ä‘á»‹nh** [10].

> **áº¨n dá»¥:** NMS giá»‘ng **kiá»ƒm tra cháº¥t lÆ°á»£ng cuá»‘i dÃ¢y chuyá»n** â€” pháº£i dá»«ng láº¡i soi tá»«ng sáº£n pháº©m, loáº¡i pháº¿ pháº©m. v10 thiáº¿t káº¿ dÃ¢y chuyá»n "zero defect" â€” khÃ´ng cáº§n kiá»ƒm tra cuá»‘i.

### 5.2.2. Consistent Dual Assignments

- **One-to-many (o2m):** 1 váº­t â†’ nhiá»u predictions â†’ nhiá»u training signal â†’ train tá»‘t
- **One-to-one (o2o):** 1 váº­t â†’ 1 prediction â†’ khÃ´ng cáº§n NMS â†’ deploy tá»‘t
- **Training:** DÃ¹ng **cáº£ 2 heads**. **Inference:** Chá»‰ dÃ¹ng **o2o head** â†’ NMS-free [10]

### 5.2.3. Káº¿t quáº£

| Model | Params | Latency | mAP@50:95 | Ghi chÃº |
|:--|:-:|:-:|:-:|:--|
| v8-S + NMS | 11.2M | 4.70ms | 44.9% | Cáº§n NMS |
| **v10-S (Dual)** | **7.2M** | **2.49ms** | **46.3%** | NMS-free, nhanh 2Ã— |
| v8-X | 68.2M | 7.61ms | 53.9% | â€” |
| **v10-X** | **29.5M** | **10.70ms** | **54.4%** | 2.3Ã— Ã­t params |

âš ï¸ **Trade-off:** v10-X (54.4%) **thua** v9-E (55.6%) vá» mAP. NhÆ°ng v10 Ã­t params gáº§n **2Ã—** vÃ  **khÃ´ng cáº§n NMS** â€” Ä‘á»•i accuracy láº¥y efficiency [9][10].

---

## 5.3. YOLO11 (2024) â€” NÃ¢ng cáº¥p toÃ n diá»‡n

**TÃ¡c giáº£:** Ultralytics | *KhÃ´ng cÃ³ paper* [11]  
*LÆ°u Ã½:* Bá» "v" trong tÃªn â€” **YOLO11** thay vÃ¬ YOLOv11.

### 5.3.1. C3K2 (CSP Bottleneck with 2 Kernel sizes)

DÃ¹ng **2 kernel sizes: $3 \times 3$ vÃ  $5 \times 5$** Ä‘á»“ng thá»i:
- Kernel $3 \times 3$: Capture **local features** (chi tiáº¿t nhá», cáº¡nh, gÃ³c)
- Kernel $5 \times 5$: Capture **wider context** (vÃ¹ng rá»™ng hÆ¡n)

> **áº¨n dá»¥:** C2f giá»‘ng nhÃ¬n qua **1 á»‘ng kÃ­nh cá»‘ Ä‘á»‹nh**. C3K2 giá»‘ng nhÃ¬n qua **2 á»‘ng kÃ­nh zoom khÃ¡c nhau cÃ¹ng lÃºc** â€” vá»«a tháº¥y chi tiáº¿t, vá»«a tháº¥y bá»‘i cáº£nh.

### 5.3.2. C2PSA (Cross Stage Partial Spatial Attention)

ThÃªm **spatial attention** vÃ o CSP â€” há»c "vÃ¹ng nÃ o quan trá»ng", highlight vÃ¹ng cÃ³ váº­t thá»ƒ, suppress background.

> **áº¨n dá»¥:** C2f giá»‘ng quÃ©t Ä‘á»u toÃ n bá»™ áº£nh (má»—i pixel xá»­ lÃ½ ngang nhau). C2PSA giá»‘ng **máº¯t ngÆ°á»i** â€” táº­p trung vÃ o vÃ¹ng quan trá»ng, lÆ°á»›t nhanh qua background.

### 5.3.3. Káº¿t quáº£

| Model | Params | mAP@50:95 | So vá»›i v8 |
|:--|:-:|:-:|:--|
| YOLO11n | 2.6M | 39.5% | v8n: 37.3% (**+2.2%**) |
| YOLO11s | 9.4M | 47.0% | v8s: 44.9% (**+2.1%**) |
| YOLO11m | 20.1M | 51.5% | v8m: 50.2% (**+1.3%**) |
| YOLO11l | 25.3M | 53.4% | v8l: 52.9% (**+0.5%**) |
| YOLO11x | 56.9M | 54.7% | v8x: 53.9% (**+0.8%**) |

---

## 5.4. YOLOv12 (2025) â€” Ká»· nguyÃªn Attention

**Paper:** "YOLOv12: Attention-Centric Real-Time Object Detectors" [12]  
**TÃ¡c giáº£:** Yunjie Tian, Qixiang Ye, David Doermann

### 5.4.1. ThÃ¡ch thá»©c

Self-attention cÃ³ complexity $O((H \times W)^2)$ â€” quÃ¡ cháº­m cho real-time. v12 giáº£i quyáº¿t báº±ng **Area Attention** [12].

> **áº¨n dá»¥:** CNN giá»‘ng Ä‘á»c sÃ¡ch **tá»«ng dÃ²ng** (local, nhanh). Attention giá»‘ng nhÃ¬n **cáº£ trang cÃ¹ng lÃºc** (global, hiá»ƒu sÃ¢u nhÆ°ng cháº­m). v12 tÃ¬m cÃ¡ch "nhÃ¬n cáº£ trang" mÃ  váº«n nhanh.

### 5.4.2. Area Attention (AÂ²)

Chia feature map thÃ nh **n vÃ¹ng báº±ng nhau**, tÃ­nh attention trong tá»«ng vÃ¹ng:

$$O(\text{Global}) = (HW)^2 \quad \rightarrow \quad O(\text{Area}, n=4) = HW \times \frac{HW}{4}$$

Giáº£m **4Ã— complexity**, chá»‰ máº¥t **0.1% AP** [12].

> **áº¨n dá»¥:** Global attention giá»‘ng 1 giÃ¡o viÃªn kiá»ƒm tra bÃ i **táº¥t cáº£** 100 há»c sinh cÃ¹ng lÃºc (quÃ¡ táº£i). Area attention chia 4 nhÃ³m 25 ngÆ°á»i, kiá»ƒm tra tá»«ng nhÃ³m â†’ nhanh 4Ã—, káº¿t quáº£ gáº§n tÆ°Æ¡ng Ä‘Æ°Æ¡ng.

### 5.4.3. R-ELAN (Residual ELAN)

**Váº¥n Ä‘á»:** Thay CNN báº±ng attention trong ELAN â†’ training báº¥t á»•n. **Giáº£i phÃ¡p:** ThÃªm residual shortcut vá»›i **scale factor 0.01** â†’ gradient á»•n Ä‘á»‹nh á»Ÿ Ä‘áº§u training [12].

### 5.4.4. Káº¿t quáº£

| Model | Params | GFLOPs | mAP@50:95 |
|:--|:-:|:-:|:-:|
| YOLO11-N | 2.6M | 6.5 | 39.5% |
| **YOLOv12-N** | **2.6M** | **6.5** | **40.6%** |
| YOLO11-S | 9.4M | 21.5 | 47.0% |
| **YOLOv12-S** | **9.3M** | **21.4** | **48.0%** |
| **YOLOv12-X** | **59.1M** | **199.0** | **55.2%** |

**Ã nghÄ©a:** v12 Ä‘Ã¡nh dáº¥u YOLO chuyá»ƒn tá»« **CNN-centric** sang **Attention-centric** [12]. NhÆ°á»£c Ä‘iá»ƒm: cáº§n GPU há»— trá»£ FlashAttention.

---

# ChÆ°Æ¡ng 6: Má»ž Rá»˜NG â€” YOLOE & YOLO26

## 6.1. YOLOE (2025) â€” Open-Vocabulary Real-Time

**Paper:** "YOLOE: Real-Time Seeing Anything" (ICCV 2025) [13]

### 6.1.1. Váº¥n Ä‘á»

Closed-set YOLO chá»‰ detect classes Ä‘Ã£ train. Open-vocabulary detectors (Grounding DINO) detect má»i thá»© nhÆ°ng quÃ¡ cháº­m. YOLOE káº¿t há»£p: **open-vocabulary + real-time** [13].

> **áº¨n dá»¥:** YOLO truyá»n thá»‘ng giá»‘ng **nhÃ¢n viÃªn báº£o vá»‡ chá»‰ biáº¿t nháº­n 80 khuÃ´n máº·t Ä‘Ã£ há»c**. YOLOE giá»‘ng báº£o vá»‡ Ä‘Æ°á»£c Ä‘Ã o táº¡o **nháº­n diá»‡n báº¥t ká»³ ai** â€” chá»‰ cáº§n báº¡n mÃ´ táº£ hoáº·c Ä‘Æ°a áº£nh máº«u.

### 6.1.2. Ba cháº¿ Ä‘á»™ Prompt

| Cháº¿ Ä‘á»™ | Input | Module | CÆ¡ cháº¿ |
|:--|:--|:--|:--|
| **Text Prompt** | "dog", "car" | RepRTA | CLIP text embeddings â†’ so khá»›p region features |
| **Visual Prompt** | áº¢nh máº«u | SAVPE | TrÃ­ch semantic + activation features â†’ cross-attention |
| **Prompt-Free** | KhÃ´ng cáº§n | LRPC | Pre-compute 1203 categories, cache â†’ similarity lookup |

**Káº¿t quáº£:** 27.1% AP trÃªn LVIS zero-shot â€” vÆ°á»£t YOLO-World-v2 (22.7%), gáº§n Grounding DINO (27.4%) nhÆ°ng **nhanh hÆ¡n ráº¥t nhiá»u** [13].

---

## 6.2. YOLO26 (2026) â€” Edge-First Design

**TÃ¡c giáº£:** Ultralytics | *KhÃ´ng cÃ³ paper chÃ­nh thá»©c* [14]  
*PhiÃªn báº£n má»›i nháº¥t tÃ­nh Ä‘áº¿n thá»i Ä‘iá»ƒm viáº¿t bÃ¡o cÃ¡o.*

### 6.2.0. Táº¡i sao gá»i lÃ  "YOLO26" thay vÃ¬ "YOLOv13"?

Tá»« YOLO11 (2024), Ultralytics Ä‘Ã£ bá» tiá»n tá»‘ "v" (version). Äáº¿n phiÃªn báº£n tiáº¿p theo, thay vÃ¬ Ä‘áº·t "YOLO12" (trÃ¹ng vá»›i YOLOv12 cá»§a nhÃ³m khÃ¡c), Ultralytics quyáº¿t Ä‘á»‹nh dÃ¹ng **quy táº¯c Ä‘áº·t tÃªn theo nÄƒm phÃ¡t hÃ nh**: YOLO**26** = phiÃªn báº£n phÃ¡t hÃ nh nÄƒm 20**26**. CÃ¡ch Ä‘áº·t tÃªn nÃ y trÃ¡nh xung Ä‘á»™t sá»‘ phiÃªn báº£n vÃ  cho tháº¥y ngay thá»i Ä‘iá»ƒm release [14].

### 6.2.1. Bá» DFL â€” Thay Ä‘á»•i triáº¿t lÃ½ lá»›n nháº¥t

DFL (Distribution Focal Loss, v8â€“v12) dá»± Ä‘oÃ¡n phÃ¢n phá»‘i 16 bins/coordinate â†’ tá»‘t accuracy nhÆ°ng **phá»©c táº¡p cho edge deployment**. YOLO26 bá» DFL, quay vá» dá»± Ä‘oÃ¡n trá»±c tiáº¿p, bÃ¹ báº±ng 3 training techniques máº¡nh hÆ¡n [14].

### 6.2.2. MuSGD (Muon-enhanced Stochastic Gradient Descent)

**Váº¥n Ä‘á»:** SGD truyá»n thá»‘ng há»™i tá»¥ cháº­m khi loss landscape phá»©c táº¡p, dá»… bá»‹ káº¹t á»Ÿ local minima.

**Giáº£i phÃ¡p:** Káº¿t há»£p SGD truyá»n thá»‘ng vá»›i **Muon optimizer** (vá»‘n Ä‘Æ°á»£c phÃ¡t triá»ƒn cho training Large Language Models). Muon sá»­ dá»¥ng **momentum trÃªn khÃ´ng gian Lie group**, cho phÃ©p cáº­p nháº­t gradient theo hÆ°á»›ng tá»‘i Æ°u hÆ¡n trÃªn manifold cá»§a trá»ng sá»‘ [14].

> **áº¨n dá»¥:** HÃ£y tÆ°á»Ÿng tÆ°á»£ng báº¡n Ä‘ang **leo nÃºi trong sÆ°Æ¡ng mÃ¹** (tá»‘i Æ°u loss). SGD truyá»n thá»‘ng giá»‘ng báº¡n chá»‰ cáº£m nháº­n Ä‘Æ°á»£c **Ä‘á»™ dá»‘c ngay dÆ°á»›i chÃ¢n** rá»“i bÆ°á»›c theo hÆ°á»›ng xuá»‘ng â€” dá»… bá»‹ káº¹t á»Ÿ thung lÅ©ng nhá». MuSGD giá»‘ng báº¡n cÃ³ thÃªm má»™t **la bÃ n chiáº¿n lÆ°á»£c** (Muon momentum) nhÃ¬n Ä‘Æ°á»£c xu hÆ°á»›ng Ä‘á»‹a hÃ¬nh rá»™ng hÆ¡n â€” khi gáº·p thung lÅ©ng nhá», la bÃ n chá»‰ báº¡n bÆ°á»›c qua vÃ¬ phÃ­a sau cÃ²n cÃ³ thung lÅ©ng sÃ¢u hÆ¡n. Káº¿t quáº£: há»™i tá»¥ nhanh hÆ¡n, accuracy cuá»‘i cÃ¹ng cao hÆ¡n.

**Hiá»‡u quáº£:** Thay tháº¿ hoÃ n toÃ n AdamW (v8â€“v12) mÃ  accuracy tÆ°Æ¡ng Ä‘Æ°Æ¡ng hoáº·c cao hÆ¡n, sá»­ dá»¥ng Ã­t memory hÆ¡n â€” phÃ¹ há»£p edge training [14].

### 6.2.3. STAL (Small-Target-Aware Label Assignment)

**Váº¥n Ä‘á»:** CÃ¡c chiáº¿n lÆ°á»£c label assignment truyá»n thá»‘ng (TAL, SimOTA) Æ°u tiÃªn váº­t lá»›n vÃ¬ IoU giá»¯a prediction vÃ  ground truth tá»± nhiÃªn cao hÆ¡n khi váº­t lá»›n. Váº­t nhá» (<8 pixels) thÆ°á»ng bá»‹ gÃ¡n Ã­t positive samples â†’ model bá» sÃ³t.

**Giáº£i phÃ¡p:** STAL Ä‘iá»u chá»‰nh **dynamic** assignment metric dá»±a trÃªn kÃ­ch thÆ°á»›c ground truth: váº­t cÃ ng nhá» â†’ threshold gÃ¡n positive Ä‘Æ°á»£c **ná»›i lá»ng** hÆ¡n â†’ nhiá»u predictions Ä‘Æ°á»£c coi lÃ  "positive" â†’ model nháº­n Ä‘Æ°á»£c nhiá»u training signal hÆ¡n cho váº­t nhá» [14].

> **áº¨n dá»¥:** Trong lá»›p há»c, **há»c sinh giá»i** (váº­t lá»›n, dá»… nháº­n ra) luÃ´n Ä‘Æ°á»£c tháº§y khen nhiá»u â€” tá»± nhiÃªn Ä‘Ã£ cÃ³ motivation. **Há»c sinh yáº¿u** (váº­t nhá», khÃ³ nháº­n ra) láº¡i Ã­t Ä‘Æ°á»£c chÃº Ã½ â†’ ngÃ y cÃ ng kÃ©m. STAL giá»‘ng tháº§y giÃ¡o **Æ°u tiÃªn Ä‘á»™ng viÃªn há»c sinh yáº¿u**: háº¡ tiÃªu chuáº©n khen thÆ°á»Ÿng cho nhÃ³m yáº¿u (ná»›i threshold), Ä‘á»ƒ cÃ¡c em nháº­n Ä‘Æ°á»£c nhiá»u feedback tÃ­ch cá»±c hÆ¡n â†’ dáº§n cáº£i thiá»‡n. Há»c sinh giá»i thÃ¬ váº«n giá»¯ tiÃªu chuáº©n cao.

**Hiá»‡u quáº£:** Äáº·c biá»‡t tá»‘t cho á»©ng dá»¥ng **drone, camera giÃ¡m sÃ¡t táº§m xa**, nÆ¡i váº­t thá»ƒ thÆ°á»ng ráº¥t nhá» trong khung hÃ¬nh [14].

### 6.2.4. ProgLoss (Progressive Loss Balancing)

**Váº¥n Ä‘á»:** YOLO cÃ³ 3 thÃ nh pháº§n loss (classification, box regression, objectness), cáº§n cÃ¢n báº±ng báº±ng cÃ¡c há»‡ sá»‘ Î». Truyá»n thá»‘ng dÃ¹ng há»‡ sá»‘ **cá»‘ Ä‘á»‹nh** suá»‘t training. NhÆ°ng thá»±c táº¿: giai Ä‘oáº¡n Ä‘áº§u model chÆ°a biáº¿t gÃ¬ â†’ cls loss ráº¥t cao â†’ chiáº¿m Æ°u tháº¿ â†’ reg loss bá»‹ "chÃ¬m"; giai Ä‘oáº¡n sau model phÃ¢n loáº¡i tá»‘t rá»“i â†’ cls loss tháº¥p â†’ láº½ ra nÃªn táº­p trung reg nhÆ°ng Î» cá»‘ Ä‘á»‹nh khÃ´ng cho phÃ©p.

**Giáº£i phÃ¡p:** ProgLoss **tá»± Ä‘á»™ng Ä‘iá»u chá»‰nh** há»‡ sá»‘ cÃ¢n báº±ng $\lambda_i$ theo training progress (epoch hiá»‡n táº¡i / tá»•ng epochs). Giai Ä‘oáº¡n Ä‘áº§u: Æ°u tiÃªn cls (cho model há»c nháº­n diá»‡n trÆ°á»›c). Giai Ä‘oáº¡n sau: dáº§n chuyá»ƒn trá»ng tÃ¢m sang reg (tinh chá»‰nh vá»‹ trÃ­) [14].

> **áº¨n dá»¥:** Giá»‘ng quÃ¡ trÃ¬nh **há»c lÃ¡i xe**. Tuáº§n Ä‘áº§u (epoch sá»›m), giÃ¡o viÃªn táº­p trung dáº¡y **nháº­n biáº¿t biá»ƒn bÃ¡o, Ä‘Ã¨n giao thÃ´ng** trÆ°á»›c (classification â€” biáº¿t cÃ¡i gÃ¬). Tuáº§n sau, khi báº¡n Ä‘Ã£ biáº¿t biá»ƒn bÃ¡o, giÃ¡o viÃªn chuyá»ƒn sang dáº¡y **cÄƒn Ä‘Æ°á»ng, Ä‘á»— xe chÃ­nh xÃ¡c** (regression â€” á»Ÿ Ä‘Ã¢u, chÃ­nh xÃ¡c cá»¡ nÃ o). Náº¿u giÃ¡o viÃªn cá»© dáº¡y song song cáº£ hai tá»« Ä‘áº§u vá»›i cÃ¹ng tá»· lá»‡ (Î» cá»‘ Ä‘á»‹nh), báº¡n sáº½ bá»‹ quÃ¡ táº£i vÃ  khÃ´ng giá»i cÃ¡i nÃ o cáº£.

**Hiá»‡u quáº£:** +0.3â€“0.5% mAP so vá»›i fixed loss weights, Ä‘áº·c biá»‡t hiá»‡u quáº£ khi bá» DFL (cáº§n reg loss máº¡nh hÆ¡n Ä‘á»ƒ bÃ¹ Ä‘áº¯p) [14].

### 6.2.5. Káº¿t quáº£

| Model | Params (Æ°á»›c tÃ­nh) | mAP@50:95 | CPU Speed | So vá»›i YOLO11 |
|:--|:-:|:-:|:--|:--|
| YOLO26-N | ~2.5M | ~40.5% | Nhanh hÆ¡n 43% | +1.0% |
| YOLO26-S | ~9M | ~47.5% | Nhanh hÆ¡n 30% | +0.5% |
| YOLO26-X | ~60M | ~57.2% | Nhanh hÆ¡n 15% | +2.5% |

*Ghi chÃº: Sá»‘ liá»‡u Æ°á»›c tÃ­nh tá»« benchmark cá»™ng Ä‘á»“ng, chÆ°a cÃ³ paper chÃ­nh thá»©c. CPU speed Ä‘o trÃªn ONNX Runtime. YOLO26-X (~57.2%) vÆ°á»£t ká»· lá»¥c mAP cá»§a v7-E6E (56.8%) â€” láº§n Ä‘áº§u tiÃªn sau 4 nÄƒm [7][14].*

### 6.2.6. Tá»•ng káº¿t triáº¿t lÃ½ YOLO26

YOLO26 Ä‘áº¡i diá»‡n cho xu hÆ°á»›ng **"less is more"**: bá» bá»›t component phá»©c táº¡p (DFL), thay báº±ng training techniques thÃ´ng minh hÆ¡n (MuSGD, STAL, ProgLoss). Káº¿t quáº£: model **nháº¹ hÆ¡n, nhanh hÆ¡n trÃªn CPU/edge**, mÃ  accuracy váº«n SOTA.
# ChÆ°Æ¡ng 7: CÃC BIáº¾N THá»‚ YOLO

## 7.1. YOLOX (2021) â€” Anchor-Free tiÃªn phong

**TÃ¡c giáº£:** Megvii | **Giáº£i thÆ°á»Ÿng:** 1st Place, CVPR 2021 Streaming Perception [15]

3 Ä‘Ã³ng gÃ³p má»Ÿ Ä‘Æ°á»ng cho YOLO sau nÃ y:

| ÄÃ³ng gÃ³p | Ã nghÄ©a | ÄÆ°á»£c adopt bá»Ÿi |
|:--|:--|:--|
| **Anchor-Free** | Chá»©ng minh YOLO hoáº¡t Ä‘á»™ng tá»‘t khÃ´ng cáº§n anchor | v6, v8+ |
| **Decoupled Head** | TÃ¡ch cls/reg thÃ nh 2 branches riÃªng biá»‡t | v6, v8, v11, v12 |
| **SimOTA** | Dynamic label assignment | Tiá»n nhiá»‡m cá»§a TAL |

YOLOX-L: 50.0% mAP, 69 FPS (V100). Nano variant: chá»‰ 0.91M params [15].

## 7.2. PP-YOLOE (2022) â€” Baidu Industrial

**TÃ¡c giáº£:** Baidu | **Framework:** PaddlePaddle [16]

**TAL (Task Alignment Learning):** Alignment metric $t = s^\alpha \times u^\beta$ Ä‘áº£m báº£o cls score cao â†” localization tá»‘t. Ká»¹ thuáº­t nÃ y Ä‘Æ°á»£c **YOLOv8, v9, v10+** adopt [16].

PP-YOLOE+-L: 53.3% mAP, 78 FPS. Phá»• biáº¿n trong cÃ´ng nghiá»‡p Trung Quá»‘c [16].

## 7.3. YOLO-NAS (2023) â€” AI thiáº¿t káº¿ AI

**TÃ¡c giáº£:** Deci AI (Israel) [17]

**AutoNAC:** KhÃ¡m phÃ¡ ~$10^{14}$ kiáº¿n trÃºc, tá»± tÃ¬m tá»‘i Æ°u. Thiáº¿t káº¿ sáºµn cho quantization: INT8 chá»‰ máº¥t **0.4â€“0.6% mAP** (cÃ¡c model khÃ¡c máº¥t 1â€“2%) [17].

## 7.4. YOLO-World (2024) â€” Tiá»n nhiá»‡m YOLOE

**TÃ¡c giáº£:** Tencent AI Lab [18]

Open-vocabulary Ä‘áº§u tiÃªn cho YOLO. **RepVL-PAN** káº¿t há»£p CLIP text embeddings vÃ o feature pyramid. Prompt-then-Detect: náº¡p text 1 láº§n, detect real-time. YOLOE sau Ä‘Ã³ cáº£i tiáº¿n thÃªm visual prompt + prompt-free [18].

---

# ChÆ°Æ¡ng 8: SO SÃNH Tá»”NG Há»¢P & ÄÃNH GIÃ

## 8.1. Báº£ng tá»•ng quan táº¥t cáº£ phiÃªn báº£n

| Version | NÄƒm | mAP (best) | Params | Anchor | NMS | Multi-Task | Paper |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| v1 [1] | 2015 | 63.4%* | ~45M | KhÃ´ng | Cáº§n | KhÃ´ng | CÃ³ |
| v2 [2] | 2016 | 78.6%* | ~50M | CÃ³ | Cáº§n | KhÃ´ng | CÃ³ |
| v3 [3] | 2018 | 33.0% | ~62M | CÃ³ | Cáº§n | KhÃ´ng | CÃ³ |
| v4 [4] | 2020 | 43.5% | ~64M | CÃ³ | Cáº§n | KhÃ´ng | CÃ³ |
| v5 [5] | 2020 | 50.7% | 86.7M | CÃ³ | Cáº§n | Háº¡n cháº¿ | KhÃ´ng |
| v6 [6] | 2022 | 52.5% | ~58M | KhÃ´ng | Cáº§n | KhÃ´ng | CÃ³ |
| v7 [7] | 2022 | 56.8% | ~71M | CÃ³ | Cáº§n | Háº¡n cháº¿ | CÃ³ |
| v8 [8] | 2023 | 53.9% | 68.2M | KhÃ´ng | Cáº§n | 6 tÃ¡c vá»¥ | KhÃ´ng |
| v9 [9] | 2024 | 55.6% | 58.1M | KhÃ´ng | Cáº§n | Háº¡n cháº¿ | CÃ³ |
| v10 [10] | 2024 | 54.4% | 29.5M | KhÃ´ng | KhÃ´ng | KhÃ´ng | CÃ³ |
| YOLO11 [11] | 2024 | 54.7% | 56.9M | KhÃ´ng | KhÃ´ng | 6 tÃ¡c vá»¥ | KhÃ´ng |
| v12 [12] | 2025 | 55.2% | 59.1M | KhÃ´ng | KhÃ´ng | CÃ³ | CÃ³ |
| YOLOE [13] | 2025 | 27.1%** | ~25M | KhÃ´ng | TÃ¹y | CÃ³ | CÃ³ |
| YOLO26 [14] | 2026 | ~57.2% | ~60M | KhÃ´ng | KhÃ´ng | 5 tÃ¡c vá»¥ | KhÃ´ng |

> *\* VOC mAP@50 â€” khÃ´ng so sÃ¡nh trá»±c tiáº¿p vá»›i COCO mAP@50:95*  
> *\*\* LVIS zero-shot â€” thang Ä‘o hoÃ n toÃ n khÃ¡c COCO closed-set*

## 8.2. Tiáº¿n hÃ³a kiáº¿n trÃºc

| ThÃ nh pháº§n | Tiáº¿n hÃ³a |
|:--|:--|
| **Backbone** | 24-layer CNN â†’ Darknet-19 â†’ Darknet-53 â†’ CSPDarknet â†’ EfficientRep â†’ E-ELAN â†’ GELAN â†’ Area Attention |
| **Anchor** | Grid (v1) â†’ Anchor-based (v2â€“v7) â†’ Anchor-free (v6, v8+) â†’ NMS-free (v10+) |
| **Loss** | MSE (v1) â†’ IoU â†’ GIoU â†’ CIoU (v4) â†’ CIoU+DFL (v8) â†’ Bá» DFL (YOLO26) |
| **Framework** | Darknet C (v1â€“v4) â†’ PyTorch `pip install` (v5+) â†’ Unified API (v8+) |

## 8.3. CÃ¡c phiÃªn báº£n "thua" Ä‘á»i trÆ°á»›c â€” PhÃ¢n tÃ­ch trade-off

| TrÆ°á»ng há»£p | Thua á»Ÿ Ä‘Ã¢u | Tháº¯ng á»Ÿ Ä‘Ã¢u |
|:--|:--|:--|
| v8 thua v7 [7][8] | mAP: 53.9% < 56.8% | Multi-task (6 tÃ¡c vá»¥), ecosystem, dá»… dÃ¹ng |
| v10 thua v9 [9][10] | mAP: 54.4% < 55.6% | Params Ã­t 2Ã—, NMS-free, latency á»•n Ä‘á»‹nh |
| v3 cháº­m hÆ¡n v2 [2][3] | FPS: 20â€“51 < 40â€“67 | Accuracy tá»‘t hÆ¡n nhiá»u (+10%) |

> **BÃ i há»c:** KhÃ´ng version nÃ o thua hoÃ n toÃ n. Má»—i "thua" lÃ  **trade-off cÃ³ chá»§ Ä‘Ã­ch** â€” hy sinh má»™t metric Ä‘á»ƒ cáº£i thiá»‡n metric khÃ¡c phÃ¹ há»£p má»¥c tiÃªu thiáº¿t káº¿.

## 8.4. Xáº¿p háº¡ng theo tiÃªu chÃ­

### Accuracy thuáº§n tÃºy (mAP@50:95 COCO)
1. YOLO26: ~57.2% [14]
2. YOLOv7-E6E: 56.8% [7]
3. YOLOv9-E: 55.6% [9]
4. YOLOv12-X: 55.2% [12]
5. YOLO11-X: 54.7% [11]

### Efficiency (mAP/Params)
YOLOv10-X dáº«n Ä‘áº§u: 54.4% mAP vá»›i chá»‰ **29.5M params** (1.84 mAP/M) [10].

### ÄÃ³ng gÃ³p lÃ½ thuyáº¿t
1. **YOLOv1** [1]: Single-stage detection â€” thay Ä‘á»•i cáº£ lÄ©nh vá»±c
2. **YOLOv9** [9]: Information Bottleneck + PGI â€” lÃ½ thuyáº¿t sÃ¢u nháº¥t
3. **YOLOv4** [4]: BoF/BoS methodology â€” há»‡ thá»‘ng hÃ³a toÃ n diá»‡n
4. **YOLOv10** [10]: NMS-free â€” giáº£i quyáº¿t váº¥n Ä‘á» tá»“n táº¡i 9 nÄƒm
5. **YOLOv12** [12]: Attention-centric â€” má»Ÿ ká»· nguyÃªn má»›i

### Production readiness (2026)
YOLO11 > YOLOv8 > YOLO26 > YOLOv12

## 8.5. SÆ¡ Ä‘á»“ pháº£ há»‡ (Family Tree)

```mermaid
flowchart TD
    subgraph Redmon["ðŸ”µ DÃ²ng Redmon (2015-2018)"]
        V1["YOLOv1<br/>2015 Â· 63.4%*"]
        V2["YOLOv2<br/>2016 Â· 78.6%*"]
        V3["YOLOv3<br/>2018 Â· 33.0%"]
    end

    subgraph Wang["ðŸŸ¢ DÃ²ng Wang (2020-2024)"]
        V4["YOLOv4<br/>2020 Â· 43.5%"]
        V7["YOLOv7<br/>2022 Â· 56.8%"]
        V9["YOLOv9<br/>2024 Â· 55.6%"]
    end

    subgraph Ultralytics["ðŸŸ  DÃ²ng Ultralytics (2020-2026)"]
        V5["YOLOv5<br/>2020 Â· 50.7%"]
        V8["YOLOv8<br/>2023 Â· 53.9%"]
        Y11["YOLO11<br/>2024 Â· 54.7%"]
        Y26["YOLO26<br/>2026 Â· ~57.2%"]
    end

    subgraph Independent["ðŸŸ£ DÃ²ng Äá»™c láº­p"]
        V6["YOLOv6<br/>2022 Â· 52.5%<br/>Meituan"]
        V10["YOLOv10<br/>2024 Â· 54.4%<br/>Tsinghua"]
        V12["YOLOv12<br/>2025 Â· 55.2%<br/>Tian et al."]
    end

    subgraph Variants["âšª Biáº¿n thá»ƒ áº£nh hÆ°á»Ÿng"]
        YOLOX["YOLOX<br/>2021 Â· Megvii"]
        PPYOLOE["PP-YOLOE<br/>2022 Â· Baidu"]
        YOLOWORLD["YOLO-World<br/>2024 Â· Tencent"]
        YOLOE["YOLOE<br/>2025 Â· ICCV"]
    end

    V1 --> V2 --> V3
    V3 -.->|"Redmon dá»«ng CV"| V4
    V4 --> V7 --> V9
    V4 --> V5 --> V8 --> Y11 --> Y26
    YOLOX -.->|"anchor-free, decoupled head"| V6
    YOLOX -.->|"decoupled head"| V8
    PPYOLOE -.->|"TAL"| V8
    V8 -.-> V10 --> V12
    YOLOWORLD -.->|"open-vocab"| YOLOE
```

## 8.6. Xu hÆ°á»›ng phÃ¡t triá»ƒn

1. **CNN â†’ Attention:** v12 má»Ÿ Ä‘áº§u, tÆ°Æ¡ng lai sáº½ hybrid CNN+Attention
2. **Closed-set â†’ Open-vocabulary:** YOLOE cho phÃ©p detect váº­t chÆ°a train [13]
3. **Server â†’ Edge:** YOLO26 tá»‘i Æ°u CPU, INT8, mobile [14]
4. **NMS â†’ NMS-free:** End-to-end detection tá»« v10 [10]
5. **CV â† NLP:** Cross-pollination tá»« LLM (MuSGD, CLIP) [14]
6. **Detection â†’ Multi-task:** 1 model cho detect+segment+pose+track [8][11]

---

# ChÆ°Æ¡ng 9: Káº¾T LUáº¬N & KHUYáº¾N NGHá»Š

## 9.1. Tá»•ng káº¿t

Qua 11 nÄƒm phÃ¡t triá»ƒn (2015â€“2026), YOLO Ä‘Ã£ tráº£i qua hÃ nh trÃ¬nh Ä‘Ã¡ng chÃº Ã½:

- **Accuracy:** Tá»« 63.4% (VOC, v1) lÃªn ~57.2% (COCO, YOLO26) â€” cáº£i thiá»‡n liÃªn tá»¥c trÃªn benchmark khÃ³ hÆ¡n nhiá»u
- **Kiáº¿n trÃºc:** Tá»« CNN 24 lá»›p thÃ´ sÆ¡ â†’ Attention-centric vá»›i Area Attention
- **Detection paradigm:** Grid â†’ Anchor â†’ Anchor-free â†’ NMS-free
- **Pháº¡m vi:** Tá»« detection only â†’ 6 tÃ¡c vá»¥ + open-vocabulary
- **Ecosystem:** Tá»« compile C code â†’ `pip install`, 1 dÃ²ng code train
- **3 dÃ²ng phÃ¡t triá»ƒn:** Redmon (v1â€“v3), Wang (v4,v7,v9), Ultralytics (v5,v8,v11,v26)

> **BÃ i há»c quan trá»ng nháº¥t:** Äá»i sau **khÃ´ng pháº£i lÃºc nÃ o cÅ©ng tá»‘t hÆ¡n** Ä‘á»i trÆ°á»›c á»Ÿ má»i tiÃªu chÃ­. Má»—i version tá»‘i Æ°u cho má»¥c Ä‘Ã­ch khÃ¡c nhau.

## 9.2. Khuyáº¿n nghá»‹ chá»n model

| Má»¥c Ä‘Ã­ch | Chá»n | LÃ½ do |
|:--|:--|:--|
| Accuracy tá»‘i Ä‘a | YOLO26-X / v7-E6E | mAP cao nháº¥t [7][14] |
| Dá»… dÃ¹ng, Ä‘a nÄƒng | v8 / YOLO11 | pip install, 6 tasks [8][11] |
| Deploy edge/mobile | YOLO26-N / v10-N | NMS-free, CPU nhanh [10][14] |
| Ãt params nháº¥t | YOLOv10-X | 29.5M cho 54.4% [10] |
| Real-time cá»±c nhanh | YOLOv6-S | 520+ FPS (TensorRT) [6] |
| Detect váº­t chÆ°a train | YOLOE | Open-vocab, 3 cháº¿ Ä‘á»™ [13] |
| Váº­t nhá» (drone) | YOLO26 | STAL chuyÃªn váº­t nhá» [14] |
| INT8 quantization | YOLO-NAS | Máº¥t chá»‰ 0.4% [17] |
| Há»c/nghiÃªn cá»©u | v1â†’v4 papers | Ná»n táº£ng lÃ½ thuyáº¿t [1]â€“[4] |

## 9.3. Lá»i cuá»‘i

> *"Sá»± tháº­t cuá»‘i cÃ¹ng: ChÃªnh lá»‡ch 1â€“3% mAP giá»¯a cÃ¡c version trÃªn COCO cÃ³ thá»ƒ khÃ´ng cÃ³ Ã½ nghÄ©a thá»‘ng kÃª trÃªn dataset thá»±c táº¿. **Data cháº¥t lÆ°á»£ng quan trá»ng hÆ¡n chá»n model.**"*

---

# PHá»¤ Lá»¤C: Báº¢NG THUáº¬T NGá»®

## Kiáº¿n trÃºc máº¡ng

| Thuáº­t ngá»¯ | Giáº£i thÃ­ch |
|:--|:--|
| **Backbone** | Pháº§n "xÆ°Æ¡ng sá»‘ng", trÃ­ch xuáº¥t features tá»« áº£nh (Darknet, CSPDarknet, ResNet) |
| **Neck** | Káº¿t ná»‘i Backbone-Head, káº¿t há»£p features nhiá»u scales (FPN, PAN) |
| **Head** | Pháº§n cuá»‘i, dá»± Ä‘oÃ¡n bounding boxes vÃ  classes |
| **CNN** | Convolutional Neural Network â€” máº¡ng dÃ¹ng phÃ©p tÃ­ch cháº­p trÃ­ch xuáº¥t features |
| **Residual Connection** | ÄÆ°á»ng táº¯t: $y = F(x) + x$. Giáº£i quyáº¿t vanishing gradient (ResNet) |
| **CSP** | Cross Stage Partial â€” chia features 2 pháº§n, giáº£m 20% computation |
| **FPN** | Feature Pyramid Network â€” káº¿t há»£p features nhiá»u scales |
| **Re-parameterization** | Training multi-branch, inference single-branch. Gá»™p weights báº±ng phÃ©p cá»™ng |
| **Attention** | CÆ¡ cháº¿ "táº­p trung" vÃ o vÃ¹ng quan trá»ng. Self-attention: global receptive field |
| **FlashAttention** | Thuáº­t toÃ¡n tÃ­nh attention tá»‘i Æ°u memory |

## Ká»¹ thuáº­t Detection

| Thuáº­t ngá»¯ | Giáº£i thÃ­ch |
|:--|:--|
| **Anchor Box** | Há»™p máº«u Ä‘á»‹nh nghÄ©a trÆ°á»›c. Model dá»± Ä‘oÃ¡n offsets. v2â€“v7 dÃ¹ng, v8+ bá» |
| **Anchor-free** | Dá»± Ä‘oÃ¡n trá»±c tiáº¿p center point + 4 distances. ÄÆ¡n giáº£n hÆ¡n |
| **NMS** | Non-Maximum Suppression â€” loáº¡i bá» boxes trÃ¹ng láº·p |
| **NMS-free** | Model tá»± output 1 box/váº­t. Tá»« v10 (dual assignment) |
| **Decoupled Head** | TÃ¡ch classification vÃ  regression thÃ nh 2 branches riÃªng |
| **Label Assignment** | Chiáº¿n lÆ°á»£c gÃ¡n prediction cho ground truth (SimOTA, TAL, STAL) |

## Metrics

| Thuáº­t ngá»¯ | Giáº£i thÃ­ch |
|:--|:--|
| **IoU** | Intersection over Union â€” Ä‘o má»©c trÃ¹ng láº·p 2 boxes (0â†’1) |
| **mAP@50** | AP á»Ÿ ngÆ°á»¡ng IoU â‰¥ 0.5. Metric chÃ­nh VOC |
| **mAP@50:95** | Trung bÃ¬nh AP á»Ÿ IoU 0.5â†’0.95. Metric chÃ­nh COCO â€” kháº¯t khe hÆ¡n |
| **FPS** | Frames Per Second. â‰¥30 FPS = real-time |
| **Params** | Sá»‘ tham sá»‘ model. 3.2M = 3.2 triá»‡u. Ãt params = model nháº¹ |
| **FLOPs** | Floating Point Operations â€” Ä‘o computational cost |

## Training & Deployment

| Thuáº­t ngá»¯ | Giáº£i thÃ­ch |
|:--|:--|
| **Mosaic** | GhÃ©p 4 áº£nh thÃ nh 1 â€” augmentation máº¡nh. v4 giá»›i thiá»‡u |
| **Quantization** | Giáº£m precision (FP32â†’INT8) â€” model nháº¹ 4Ã—, cÃ³ thá»ƒ máº¥t accuracy |
| **BoF/BoS** | Bag of Freebies/Specials â€” ká»¹ thuáº­t tÄƒng accuracy miá»…n phÃ­/cÃ³ phÃ­ nháº¹ |
| **TensorRT** | NVIDIA SDK tá»‘i Æ°u inference trÃªn GPU. TÄƒng 2â€“5Ã— speed |
| **ONNX** | Format chuáº©n export model, cháº¡y trÃªn nhiá»u hardware |

---

# TÃ€I LIá»†U THAM KHáº¢O

| # | TÃ i liá»‡u |
|:-:|:--|
| [1] | J. Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection," CVPR 2016. arXiv:1506.02640 |
| [2] | J. Redmon, A. Farhadi, "YOLO9000: Better, Faster, Stronger," CVPR 2017. arXiv:1612.08242 |
| [3] | J. Redmon, A. Farhadi, "YOLOv3: An Incremental Improvement," 2018. arXiv:1804.02767 |
| [4] | A. Bochkovskiy et al., "YOLOv4: Optimal Speed and Accuracy of Object Detection," 2020. arXiv:2004.10934 |
| [5] | G. Jocher, "YOLOv5," Ultralytics, 2020. github.com/ultralytics/yolov5 |
| [6] | C. Li et al., "YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications," 2022. arXiv:2209.02976 |
| [7] | C.-Y. Wang et al., "YOLOv7: Trainable Bag-of-Freebies," CVPR 2023. arXiv:2207.02696 |
| [8] | G. Jocher, "YOLOv8," Ultralytics, 2023. github.com/ultralytics/ultralytics |
| [9] | C.-Y. Wang et al., "YOLOv9: Learning What You Want to Learn Using PGI," 2024. arXiv:2402.13616 |
| [10] | A. Wang et al., "YOLOv10: Real-Time End-to-End Object Detection," 2024. arXiv:2405.14458 |
| [11] | G. Jocher, "YOLO11," Ultralytics, 2024. github.com/ultralytics/ultralytics |
| [12] | Y. Tian et al., "YOLOv12: Attention-Centric Real-Time Object Detectors," 2025. arXiv:2502.12524 |
| [13] | "YOLOE: Real-Time Seeing Anything," ICCV 2025. arXiv:2503.07465 |
| [14] | G. Jocher, "YOLO26," Ultralytics, 2026. github.com/ultralytics/ultralytics |
| [15] | Z. Ge et al., "YOLOX: Exceeding YOLO Series in 2021," 2021. arXiv:2107.08430 |
| [16] | S. Xu et al., "PP-YOLOE: An Evolved Version of YOLO," 2022. arXiv:2203.16250 |
| [17] | "YOLO-NAS," Deci AI, 2023. github.com/Deci-AI/super-gradients |
| [18] | T. Cheng et al., "YOLO-World: Real-Time Open-Vocabulary Object Detection," 2024. arXiv:2401.17270 |
