# DỊCH CHI TIẾT PAPER YOLOv1 — Phần 1/2
**"You Only Look Once: Unified, Real-Time Object Detection"**
**Tác giả:** Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
**Hội nghị:** CVPR 2016 | **Link:** https://arxiv.org/abs/1506.02640

---

## ABSTRACT (Tóm tắt)

Chúng tôi trình bày YOLO, một cách tiếp cận mới cho bài toán phát hiện vật thể (object detection). Các nghiên cứu trước đây tận dụng lại các bộ phân loại (classifiers) để thực hiện detection. Thay vào đó, chúng tôi xây dựng object detection như một **bài toán hồi quy** (regression problem) tới các bounding boxes được phân tách theo không gian và các xác suất lớp (class probabilities) tương ứng. Một mạng neural duy nhất dự đoán bounding boxes và class probabilities trực tiếp từ toàn bộ ảnh chỉ trong **một lần đánh giá duy nhất**. Vì toàn bộ pipeline detection là một mạng duy nhất, nó có thể được tối ưu end-to-end trực tiếp trên hiệu suất detection.

Kiến trúc thống nhất của chúng tôi cực kỳ nhanh. Model YOLO cơ bản xử lý ảnh real-time ở **45 frames/giây**. Phiên bản nhỏ hơn, Fast YOLO, xử lý **155 frames/giây** mà vẫn đạt gấp đôi mAP so với các detector real-time khác. So với các hệ thống detection tiên tiến nhất (SOTA), YOLO mắc nhiều lỗi localization hơn nhưng **ít bị false positive trên background** hơn rất nhiều. Cuối cùng, YOLO học được các biểu diễn rất tổng quát về vật thể. Nó vượt trội tất cả phương pháp detection khác, bao gồm DPM và R-CNN, khi tổng quát hóa từ ảnh tự nhiên sang nghệ thuật trên cả Picasso Dataset và People-Art Dataset.

---

## 1. INTRODUCTION (Giới thiệu)

Con người nhìn lướt qua một ảnh và ngay lập tức biết có vật gì trong ảnh, chúng ở đâu, và chúng tương tác với nhau thế nào. Hệ thống thị giác của con người nhanh và chính xác, cho phép chúng ta thực hiện các tác vụ phức tạp như lái xe mà không cần suy nghĩ có ý thức. Các thuật toán detection nhanh và chính xác sẽ cho phép máy tính lái xe mà không cần cảm biến chuyên dụng, cho phép thiết bị hỗ trợ truyền tải thông tin cảnh vật theo thời gian thực cho người dùng, và mở ra tiềm năng cho hệ thống robot đa năng, phản ứng nhanh.

Các hệ thống detection hiện tại tận dụng lại classifiers để thực hiện detection. Để detect một vật thể, các hệ thống này lấy classifier cho vật thể đó và đánh giá nó tại nhiều vị trí và tỷ lệ khác nhau trên ảnh. Các hệ thống như Deformable Parts Models (DPM) sử dụng sliding window, trong đó classifier được chạy tại các vị trí được phân bố đều trên toàn bộ ảnh.

Các cách tiếp cận gần đây hơn như R-CNN sử dụng phương pháp đề xuất vùng (region proposal): đầu tiên tạo ra các bounding boxes tiềm năng trong ảnh, sau đó chạy classifier trên các boxes đề xuất này. Sau classification, hậu xử lý (post-processing) được dùng để tinh chỉnh bounding boxes, loại bỏ detection trùng lặp, và tái đánh giá boxes dựa trên các vật thể khác trong cảnh. Các pipeline phức tạp này chậm và khó tối ưu vì mỗi thành phần riêng biệt phải được train riêng.

Chúng tôi coi bài toán object detection là một bài toán hồi quy duy nhất, đi thẳng từ pixel ảnh tới tọa độ bounding box và class probabilities. Sử dụng hệ thống của chúng tôi, bạn chỉ cần nhìn (You Only Look Once — YOLO) vào ảnh để dự đoán có vật thể gì và chúng ở đâu.

YOLO cực kỳ đơn giản: xem Hình 1. Một mạng convolutional duy nhất đồng thời dự đoán nhiều bounding boxes và class probabilities cho các boxes đó. YOLO train trên toàn bộ ảnh và trực tiếp tối ưu hiệu suất detection. Mô hình thống nhất này có nhiều ưu điểm so với các phương pháp detection truyền thống.

**Thứ nhất, YOLO cực kỳ nhanh.** Vì chúng tôi mô hình detection như bài toán hồi quy, nên không cần pipeline phức tạp. Chúng tôi chỉ đơn giản chạy mạng neural trên ảnh mới tại thời điểm test để dự đoán. Model cơ bản chạy 45 FPS trên GPU Titan X và phiên bản nhanh chạy hơn 150 FPS. Điều này có nghĩa chúng tôi có thể xử lý video streaming real-time với độ trễ dưới 25 mili-giây. Hơn nữa, YOLO đạt hơn gấp đôi mAP so với các hệ thống real-time khác.

**Thứ hai, YOLO suy luận toàn cục về ảnh khi đưa ra dự đoán.** Không giống sliding window và region proposal, YOLO nhìn toàn bộ ảnh trong quá trình training và test, do đó nó mã hóa ngầm thông tin ngữ cảnh về các lớp cũng như hình dạng của chúng. Fast R-CNN, phương pháp detection hàng đầu, nhầm vùng background thành vật thể vì nó không thể nhìn thấy ngữ cảnh rộng hơn. YOLO mắc ít hơn một nửa số lỗi background so với Fast R-CNN.

**Thứ ba, YOLO học các biểu diễn tổng quát hóa được của vật thể.** Khi train trên ảnh tự nhiên và test trên nghệ thuật, YOLO vượt trội xa các phương pháp detection hàng đầu như DPM và R-CNN. Vì YOLO có khả năng tổng quát hóa cao, nó ít bị sụp đổ khi áp dụng vào domain hoặc input bất ngờ.

YOLO vẫn tụt hậu so với các hệ thống detection SOTA về accuracy. Mặc dù có thể nhanh chóng nhận diện vật thể trong ảnh, nó gặp khó khăn trong việc định vị chính xác một số vật thể, đặc biệt là vật nhỏ.

---

## 2. UNIFIED DETECTION (Phát hiện thống nhất)

Chúng tôi thống nhất các thành phần riêng biệt của object detection vào một mạng neural duy nhất. Mạng của chúng tôi sử dụng features từ toàn bộ ảnh để dự đoán mỗi bounding box. Nó cũng dự đoán tất cả bounding boxes trên tất cả classes cho một ảnh đồng thời. Điều này có nghĩa mạng suy luận toàn cục về toàn bộ ảnh và tất cả vật thể trong ảnh. Thiết kế YOLO cho phép training end-to-end và tốc độ real-time trong khi duy trì mAP cao.

Hệ thống chia ảnh đầu vào thành lưới **S × S**. Nếu tâm của một vật thể rơi vào một ô lưới, ô lưới đó chịu trách nhiệm phát hiện vật thể đó.

Mỗi ô lưới dự đoán **B** bounding boxes và **confidence scores** cho các boxes đó. Confidence scores phản ánh mức độ tự tin rằng box chứa vật thể và box đó chính xác đến mức nào. Chúng tôi định nghĩa confidence = **Pr(Object) × IoU(pred, truth)**. Nếu không có vật thể nào trong ô, confidence scores = 0. Nếu có, chúng tôi muốn confidence score bằng IoU giữa predicted box và ground truth.

Mỗi bounding box gồm 5 dự đoán: **x, y, w, h, confidence**.
- **(x, y)** biểu diễn tâm của box tương đối so với biên của ô lưới
- **w, h** được dự đoán tương đối so với toàn bộ ảnh
- **confidence** biểu diễn IoU giữa predicted box và bất kỳ ground truth box nào

Mỗi ô lưới cũng dự đoán **C** conditional class probabilities: **Pr(Class_i | Object)**. Các xác suất này được điều kiện hóa trên ô lưới chứa vật thể. Chúng tôi chỉ dự đoán một tập class probabilities cho mỗi ô lưới, bất kể số lượng boxes B.

Tại thời điểm test, nhân conditional class probabilities với confidence predictions:

> **Pr(Class_i | Object) × Pr(Object) × IoU = Pr(Class_i) × IoU**

Tích này cho class-specific confidence scores cho mỗi box. Scores này mã hóa cả xác suất lớp đó xuất hiện trong box VÀ box khớp với vật thể tốt đến mức nào.

Để đánh giá trên PASCAL VOC: S = 7, B = 2, C = 20.
→ Tensor đầu ra: **7 × 7 × 30**

---

## 2.1. NETWORK DESIGN (Thiết kế mạng)

Mạng được triển khai như mạng convolutional. Các lớp convolutional đầu trích xuất features từ ảnh, trong khi các lớp fully connected dự đoán output probabilities và coordinates.

Kiến trúc mạng lấy cảm hứng từ model GoogLeNet cho image classification. Mạng có **24 lớp convolutional** tiếp theo **2 lớp fully connected**. Thay vì inception modules của GoogLeNet, chúng tôi sử dụng các lớp reduction 1×1 tiếp theo lớp convolutional 3×3, tương tự Lin et al.

Chúng tôi cũng train phiên bản nhanh, **Fast YOLO**, sử dụng mạng neural với ít lớp conv hơn (**9 thay vì 24**) và ít filters hơn trong các lớp đó. Ngoài kích thước mạng, tất cả training và testing parameters giống nhau giữa YOLO và Fast YOLO.

Output cuối cùng: tensor **7 × 7 × 30** predictions.

---

## 2.2. TRAINING (Huấn luyện)

**Pre-training:** 20 lớp conv đầu tiên trên ImageNet 1000-class competition dataset. Cho pre-training, dùng 20 lớp conv + average-pooling + fully connected layer. Train khoảng 1 tuần, đạt top-5 accuracy 88% trên ImageNet 2012 validation (tương đương GoogLeNet). Dùng framework Darknet cho training và inference.

**Chuyển sang detection:** Thêm 4 lớp conv + 2 lớp FC với random weights. Detection thường yêu cầu thông tin thị giác chi tiết nên tăng input resolution từ **224×224 lên 448×448**.

Lớp cuối dự đoán cả class probabilities và bounding box coordinates. Chuẩn hóa w, h bằng chiều rộng/cao ảnh → nằm trong [0, 1]. Parameterize x, y là offsets so với ô lưới → cũng nằm trong [0, 1].

**Hàm kích hoạt:** Leaky ReLU trên tất cả lớp trừ lớp cuối (linear activation):
> φ(x) = x nếu x > 0; 0.1x nếu x ≤ 0

**Hàm Loss** — Tối ưu bằng sum-squared error (SSE). Tuy nhiên, SSE có vấn đề:

1. **Localization error vs classification error:** SSE coi chúng ngang hàng, nhưng không lý tưởng. Dùng **λ_coord = 5** để tăng trọng số loss từ bounding box coordinates.

2. **Mất cân bằng object/no-object:** Nhiều ô không chứa vật thể → confidence bị đẩy về 0, lấn át gradient từ ô có vật → model bất ổn, phân kỳ sớm. Dùng **λ_noobj = 0.5** để giảm trọng số loss từ confidence cho ô không có vật.

3. **Box lớn vs box nhỏ:** SSE coi sai số ở box lớn và nhỏ ngang nhau, nhưng sai lệch nhỏ ở box nhỏ quan trọng hơn. Giải pháp: dự đoán **căn bậc 2** của width và height (thay vì trực tiếp).

YOLO dự đoán nhiều bounding boxes mỗi ô. Tại training time, chỉ 1 predictor "chịu trách nhiệm" (responsible) cho mỗi vật thể — là predictor có IoU cao nhất với ground truth. Điều này dẫn đến chuyên môn hóa giữa các bounding box predictors: mỗi predictor trở nên tốt hơn ở dự đoán kích thước, aspect ratios, hoặc classes nhất định.

**Hàm loss đầy đủ** (5 thành phần):
1. Sai số tọa độ tâm (x, y) — chỉ ô có object, λ_coord = 5
2. Sai số kích thước (√w, √h) — chỉ ô có object, λ_coord = 5
3. Confidence loss — ô có object
4. Confidence loss — ô không có object, λ_noobj = 0.5
5. Classification loss — chỉ ô có object

**Training details:**
- Train trên PASCAL VOC 2007 + 2012 training/validation data
- 135 epochs. Batch size 64, momentum 0.9, decay 0.0005
- Learning rate schedule: tăng dần từ 10⁻³ lên 10⁻² (epoch đầu), giữ 10⁻² (75 epochs), giảm 10⁻³ (30 epochs), giảm 10⁻⁴ (30 epochs)
- Dropout: rate 0.5 sau lớp FC đầu tiên
- Data augmentation: random scaling/translations (lên đến 20% ảnh), random chỉnh exposure/saturation (hệ số đến 1.5 trong không gian HSV)

## 2.3. INFERENCE (Suy luận)

Giống training, dự đoán detections trên ảnh test chỉ cần **một lần đánh giá mạng duy nhất**. Trên PASCAL VOC, mạng dự đoán 98 bounding boxes cho mỗi ảnh (7×7 ô × 2 boxes/ô) cùng class probabilities cho mỗi box. YOLO cực kỳ nhanh tại thời điểm test vì chỉ cần một lần forward pass qua mạng, không giống các phương pháp dựa trên classifier.

Hệ thống lưới buộc phải có sự đa dạng không gian trong các bounding box predictions. Thường thì rõ ràng vật thể rơi vào ô lưới nào, và mạng chỉ dự đoán một box cho mỗi vật thể. Tuy nhiên, một số vật thể lớn hoặc vật thể gần biên của nhiều ô có thể được localize tốt bởi nhiều ô. **Non-maximal suppression (NMS)** có thể được sử dụng để xử lý các detection đa lớp. NMS cải thiện mAP thêm 2-3%.

---

## 2.4. LIMITATIONS OF YOLO (Hạn chế)

YOLO áp đặt **ràng buộc không gian mạnh** lên bounding box predictions vì mỗi ô lưới chỉ dự đoán 2 boxes và chỉ có thể có 1 class. Ràng buộc này giới hạn số lượng vật thể gần nhau mà model có thể dự đoán. Model gặp khó khăn với **nhóm vật thể nhỏ** xuất hiện theo cụm, ví dụ đàn chim.

Vì model học dự đoán bounding boxes từ data, nó gặp khó khăn trong việc tổng quát hóa cho vật thể với **aspect ratios hoặc cấu hình mới/bất thường**. Model cũng sử dụng features tương đối thô để dự đoán bounding boxes vì kiến trúc có nhiều lớp downsampling từ ảnh đầu vào.

Cuối cùng, trong khi train hàm loss xấp xỉ hiệu suất detection, hàm loss **coi sai số ngang nhau** ở bounding boxes nhỏ và lớn. Sai lệch nhỏ ở box lớn thường không đáng kể, nhưng sai lệch nhỏ ở box nhỏ ảnh hưởng rất lớn đến IoU. Nguồn lỗi chính là **localization không chính xác**.

---

## 3. COMPARISON TO OTHER DETECTION SYSTEMS (So sánh)

### Deformable Parts Models (DPM)
DPM sử dụng sliding window cho object detection. DPM dùng pipeline rời rạc: trích xuất static features, phân loại vùng, dự đoán bounding boxes cho vùng điểm cao, v.v. YOLO thay thế tất cả các phần rời rạc này bằng một mạng neural duy nhất. Mạng thực hiện đồng thời trích xuất features, dự đoán bounding boxes, non-maximal suppression, và contextual reasoning. Thay vì static features, mạng train features in-line và tối ưu chúng cho tác vụ detection. Kiến trúc thống nhất dẫn đến model nhanh hơn, chính xác hơn DPM.

### R-CNN
R-CNN và các biến thể sử dụng region proposals thay vì sliding windows. Selective Search tạo potential bounding boxes, mạng convolutional trích xuất features, SVM phân loại boxes, linear model điều chỉnh bounding boxes, và NMS loại bỏ duplicate detections. Mỗi giai đoạn phải được fine-tuned độc lập, hệ thống rất chậm (mất hơn 40 giây/ảnh).

YOLO chia sẻ một số điểm tương đồng với R-CNN. Mỗi ô lưới đề xuất potential bounding boxes và cho điểm bằng convolutional features. Tuy nhiên, hệ thống YOLO đặt ràng buộc không gian lên grid cell proposals, giúp giảm thiểu duplicate detections. YOLO cũng đề xuất ít bounding boxes hơn nhiều: chỉ 98/ảnh so với khoảng 2000 của Selective Search. Cuối cùng, YOLO kết hợp các thành phần riêng lẻ thành một model tối ưu chung.

### Các cách tiếp cận khác
- **Deep MultiBox:** train mạng convolutional dự đoán regions of interest thay vì Selective Search. Tuy nhiên không thể thực hiện general object detection.
- **OverFeat:** train CNN dự đoán localization với tối ưu cho localization thay vì detection. OverFeat vẫn là hệ thống rời rạc, tối ưu cho localization chứ không phải detection performance.
- **MultiGrasp:** Hệ thống YOLO tương tự thiết kế grasp detection. Tuy nhiên grasp detection đơn giản hơn nhiều vì chỉ cần dự đoán 1 graspable region cho 1 vật thể, không cần dự đoán kích thước, vị trí, biên, class.

---

## 4. EXPERIMENTS (Thực nghiệm)

### 4.1. So sánh với hệ thống Real-Time khác

Nhiều nghiên cứu cố gắng tăng tốc pipeline detection. Tuy nhiên, chỉ có Sadeghi et al. thực sự đạt detection real-time (30 FPS). So sánh trên PASCAL VOC 2007:

| Phương pháp | mAP | FPS |
|------------|-----|-----|
| 100Hz DPM | 16.0% | 100 |
| 30Hz DPM | 26.1% | 30 |
| **Fast YOLO** | **52.7%** | **155** |
| **YOLO** | **63.4%** | **45** |
| YOLO VGG-16 | 66.4% | 21 |

Fast YOLO là detector real-time nhanh nhất trên PASCAL VOC và mAP gấp đôi bất kỳ phương pháp real-time nào khác. YOLO đẩy mAP lên 63.4% trong khi vẫn duy trì real-time.

Khi dùng VGG-16 làm backbone (thay vì Darknet): mAP tăng lên 66.4% nhưng chậm hơn đáng kể. Các thí nghiệm còn lại dùng Darknet vì nhanh hơn.

### 4.2. VOC 2007 Error Analysis

So sánh YOLO với Fast R-CNN (một trong những phương pháp tốt nhất). Phân tích sử dụng phương pháp của Hoiem et al., phân loại mỗi detection:
- **Correct:** Class đúng, IoU > 0.5
- **Localization:** Class đúng, 0.1 < IoU < 0.5
- **Similar:** Class tương tự, IoU > 0.1
- **Other:** Class sai, IoU > 0.1
- **Background:** IoU < 0.1 với bất kỳ vật nào

**Kết quả phân tích lỗi:**

| Loại lỗi | YOLO | Fast R-CNN |
|----------|------|-----------|
| Correct | 71.6% | 71.6% |
| **Localization** | **19.0%** | 8.6% |
| Similar | 4.3% | 4.1% |
| Other | 1.7% | 6.8% |
| **Background** | 3.4% | **8.9%** |

**Nhận xét quan trọng:**
- YOLO gặp khó khăn nhiều hơn trong việc localize vật thể chính xác. Lỗi localization chiếm tỷ lệ lớn hơn tất cả lỗi khác cộng lại.
- Fast R-CNN mắc lỗi background nhiều hơn gần 3 lần. 13.6% top detections là false positives không chứa vật thể. Fast R-CNN predict false positives trên background nhiều gấp gần 3 lần YOLO.

### 4.3. Kết hợp Fast R-CNN và YOLO

Vì YOLO mắc ít lỗi background hơn nhiều so với Fast R-CNN, dùng YOLO để loại bỏ false positive detections từ Fast R-CNN giúp cải thiện đáng kể. Với mỗi bounding box mà R-CNN dự đoán, kiểm tra xem YOLO có dự đoán box tương tự không. Nếu có → giữ, tăng confidence.

Kết quả:
- Fast R-CNN alone: 71.8% mAP
- **Fast R-CNN + YOLO: 75.0% mAP** (tăng 3.2%)
- Fast R-CNN + các model Fast R-CNN khác: chỉ cải thiện ít (0.3-0.6%) → YOLO mang lại cải thiện nhiều hơn vì mắc lỗi **khác loại** so với R-CNN

Kết hợp model không tăng tốc YOLO vì chạy từng model riêng rồi combine kết quả. Nhưng YOLO rất nhanh nên không thêm đáng kể computational time so với Fast R-CNN.

### 4.4. VOC 2012 Results

Trên PASCAL VOC 2012 test set, YOLO đạt **57.9% mAP**. Thấp hơn SOTA (R-CNN với VGG + re-ranking) nhưng tốc độ nhanh hơn rất nhiều. Sử dụng kết hợp Fast R-CNN + YOLO đạt kết quả cạnh tranh.

YOLO yếu nhất ở các lớp: bottle, sheep, tv/monitor (vật nhỏ).
YOLO mạnh nhất ở: cat, train, person (vật lớn, rõ ràng).

### 4.5. Generalizability (Khả năng tổng quát hóa)

Test YOLO trên 2 bộ dữ liệu nghệ thuật: **Picasso Dataset** và **People-Art Dataset** — domain hoàn toàn khác so với training data (ảnh tự nhiên).

| Phương pháp | VOC 2007 | Picasso | People-Art |
|------------|---------|---------|-----------|
| R-CNN | 54.2% | 10.2% | 21% |
| DPM | 43.2% | 32.4% | 25% |
| DPM (Person) | - | 37.8% | 26% |
| **YOLO** | **59.2%** | **53.3%** | **45%** |

**YOLO vượt trội rất xa khi generalize sang domain mới.** Điều này cho thấy YOLO học được biểu diễn tổng quát của vật thể, không chỉ ghi nhớ patterns cụ thể từ training data. R-CNN sử dụng Selective Search cho region proposals — các proposals này được tune cho ảnh tự nhiên nên thất bại trên artwork.

---

## 5. REAL-TIME DETECTION IN THE WILD (Detection thực tế)

YOLO là detector nhanh, chính xác → lý tưởng cho CV applications. Khi kết nối với webcam, hoạt động như tracking system, phát hiện vật thể khi chúng di chuyển và hình dạng thay đổi. Demo có tại trang web dự án: http://pjreddie.com/yolo/

---

## 6. CONCLUSION (Kết luận)

Chúng tôi giới thiệu YOLO, một model thống nhất cho object detection. Model được xây dựng đơn giản và có thể train trực tiếp trên toàn bộ ảnh. Không giống các cách tiếp cận dựa trên classifier, YOLO được train trên hàm loss tương ứng trực tiếp với hiệu suất detection và toàn bộ model được train cùng lúc.

Fast YOLO là detector đa năng nhanh nhất trong tài liệu và YOLO đẩy SOTA cho object detection real-time. YOLO cũng tổng quát hóa tốt sang domain mới, làm cho nó lý tưởng cho các ứng dụng phụ thuộc vào detection nhanh, robust.

---

## BẢNG KẾT QUẢ CHI TIẾT TỪNG CLASS — PASCAL VOC 2007

| Class | YOLO | Fast R-CNN | Faster R-CNN |
|-------|------|-----------|-------------|
| aero | 77.0 | 77.0 | 76.5 |
| bike | 67.2 | 78.1 | 79.0 |
| bird | 57.7 | 68.1 | 70.9 |
| boat | 38.3 | 52.6 | 45.6 |
| bottle | 22.7 | 34.4 | 27.3 |
| bus | 68.3 | 75.9 | 70.8 |
| car | 55.9 | 76.0 | 79.8 |
| cat | 81.4 | 86.4 | 79.6 |
| chair | 36.2 | 47.6 | 38.1 |
| cow | 60.8 | 68.4 | 73.6 |
| table | 48.5 | 59.3 | 62.2 |
| dog | 77.2 | 81.5 | 75.2 |
| horse | 72.3 | 78.0 | 76.7 |
| mbike | 71.3 | 75.7 | 71.7 |
| person | 63.5 | 72.5 | 78.1 |
| plant | 28.9 | 37.6 | 34.2 |
| sheep | 52.2 | 62.3 | 63.4 |
| sofa | 54.8 | 65.5 | 55.8 |
| train | 73.9 | 79.6 | 77.0 |
| tv | 50.8 | 56.3 | 49.0 |
| **mAP** | **63.4** | **70.0** | **73.2** |

---

## TÀI LIỆU THAM KHẢO TRONG PAPER

Paper trích dẫn 27 tài liệu, bao gồm:
1. DPM (Felzenszwalb et al., 2010) — Sliding window detection
2. R-CNN (Girshick et al., 2014) — Region-based CNN
3. Fast R-CNN (Girshick, 2015) — Cải tiến R-CNN
4. GoogLeNet (Szegedy et al., 2014) — Cảm hứng kiến trúc
5. ImageNet (Russakovsky et al., 2015) — Pre-training dataset
6. Selective Search (Uijlings et al., 2013) — Region proposals
7. Network in Network (Lin et al., 2013) — 1×1 convolutions
8. OverFeat (Sermanet et al., 2013) — Integrated recognition
9. MultiBox (Erhan et al., 2014) — Bounding box prediction
10. Batch Normalization (Ioffe & Szegedy, 2015)
11. VGGNet (Simonyan & Zisserman, 2014)
12. PASCAL VOC (Everingham et al., 2010) — Benchmark dataset

---

*Bản dịch chi tiết từ paper gốc arXiv:1506.02640. Mọi nội dung kỹ thuật giữ nguyên ý nghĩa.*

