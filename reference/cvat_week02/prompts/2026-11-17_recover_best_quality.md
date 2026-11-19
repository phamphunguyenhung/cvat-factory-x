# RECOVER BEST ANNOTATION QUALITY — WEEK 2 G06

> Prompt đã lưu ngày 17/11/2026 để thực hiện sau. Chưa chạy.

## ROLE: Principal CV Engineer + Annotation Recovery Lead

Bạn đang tiếp tục dự án tại:

`D:\06-Learning\AIA-VINGROUP\learning\cvat`

**ĐÂY LÀ NHIỆM VỤ KHÔI PHỤC CHẤT LƯỢNG ANNOTATION, KHÔNG PHẢI NHIỆM VỤ CHẠY LẠI MODEL.**

Người dùng phát hiện kết quả CVAT hiện tại có chất lượng kém hơn phiên làm việc buổi sáng 17/11/2026.

Đặc biệt, job 1975 có các điểm mắt, lông mày, môi bị đặt sai hoặc dồn về vùng không phù hợp.

Nguyên nhân nghi ngờ: các lần chạy sau đã thay thế hoặc bỏ qua kết quả tinh chỉnh trước đó, chỉ sử dụng nháp từ local model.

**MỤC TIÊU: Khôi phục chất lượng tốt nhất đã đạt được, tiếp tục cải thiện từng ảnh và đồng bộ các sửa đổi tọa độ thực sự lên CVAT.**

---

## 1. TRUY VẾT LỊCH SỬ PHIÊN SÁNG — BẮT BUỘC

Trước khi chạy bất kỳ model nào:

1. Đọc toàn bộ lịch sử Claude Code có thể truy cập được, đặc biệt các phiên ngày 17/11/2026.
2. Tìm các lần người dùng đã hướng dẫn sửa mắt, môi, lông mày và các keypoint bị lệch.
3. Đọc memory bank, CLAUDE.md, guidelines, issues.md.
4. Kiểm tra Git history, reflog, local checkpoints, reports, backups và các file annotation JSON.
5. Đối chiếu các bản annotation đã tồn tại theo thời gian.
6. Xác định các script và công cụ đã được tối ưu trong phiên sáng.
7. Phân biệt rõ kết quả model gốc, kết quả đã tinh chỉnh bằng tools, và những điểm thực sự được người dùng sửa hoặc xác nhận.

Không được tuyên bố đã đọc lịch sử nếu thực tế không truy cập được. Nếu lịch sử chat không đầy đủ, sử dụng checkpoints và backups làm nguồn bổ sung, ghi rõ phần còn thiếu.

Tạo:

`reports/w2_recovery/RECOVERY_TIMELINE.md`

Bao gồm:

- Thời điểm của từng phiên bản annotation.
- Model và tools được sử dụng.
- Những frame được chỉnh sửa.
- Các điểm thay đổi tọa độ.
- Chất lượng và bằng chứng so sánh.
- Khả năng một bản nháp cũ đã ghi đè kết quả tốt hơn.

**Không restore toàn bộ bản cũ một cách mù quáng. Phải so sánh từng frame và từng keypoint.**

---

## 2. KHÔI PHỤC PIPELINE TỐT NHẤT

Đọc hai guideline BTC:

`C:\Users\ppnh1\Downloads\HumanPose17_Annotation_Guideline_V1.docx`

`C:\Users\ppnh1\Downloads\Face_Landmark50_Annotation_Guideline_V1.docx`

Bản BTC là nguồn chuẩn cao nhất khi xác định vị trí và trạng thái keypoint.

Tìm lại pipeline đã tối ưu từ phiên sáng.

Pipeline mong muốn:

Original Image
→ Existing Best Annotation
→ Detect Difficult Cases
→ Haiku Visual Analysis
→ Identify Anatomical Landmarks
→ Propose Coordinate Corrections
→ Apply Local Geometry Refinement
→ Render Before/After
→ Independent Haiku QC
→ CVAT Sync
→ Read-back Verification

Local model chỉ được dùng để cung cấp ứng viên vị trí, không được tự động thay thế bản annotation đã tinh chỉnh tốt hơn.

Đặc biệt:

**KHÔNG ĐƯỢC THAY BẢN ĐÃ SỬA BẰNG OUTPUT THÔ CỦA MODEL.**

Trước khi sửa phải tạo backup bất biến của trạng thái server và local.

---

## 3. PHÂN CÔNG HAIKU AGENTS THEO CHỨC NĂNG

Sử dụng Claude Code Task agents chạy model Haiku trực tiếp theo khả năng hiện có.

Không yêu cầu người dùng cung cấp Anthropic API key riêng.

### Agent A — History Recovery

- Truy tìm các phiên bản annotation cũ.
- Tìm các tọa độ từng được cải thiện.
- So sánh điểm trước và sau.
- Xác định phiên bản tốt nhất có bằng chứng cho từng frame.
- Báo cáo các dấu hiệu regression.

Output: `recovery_candidates.json`

### Agent B — Face Anatomy Specialist

Phụ trách job 1975.

Không chỉ nói "môi lệch" hoặc "mắt sai".

Phải trực tiếp phân tích ảnh gốc và crop đủ độ phân giải để nhận biết:

- Cung lông mày trái/phải.
- Mí trên, mí dưới và khóe mắt.
- Trục mắt khi đầu nghiêng.
- Đường viền môi ngoài.
- Đường viền môi trong.
- Hai khóe miệng.
- Vùng bị che bởi kính, tay, râu.
- Các landmark không thể nhìn thấy.

Với mỗi điểm sai phải cung cấp tọa độ đề xuất hoặc trạng thái occluded/outside phù hợp guideline.

Không được tự suy ra cấu trúc giải phẫu khi ảnh không đủ thông tin.

### Agent C — Pose Anatomy Specialist

Phụ trách job 1623.

Kiểm tra:

- Đầu, mắt, tai.
- Vai, khuỷu tay, cổ tay.
- Hông, đầu gối, cổ chân.
- Trái/phải theo guideline BTC.
- Occlusion và outside.
- Người nằm nghiêng, xoay đầu hoặc bị vật che.

Không dùng phép so sánh tọa độ X đơn thuần để quyết định trái/phải khi ảnh bị xoay.

### Agent D — Geometry Refiner

- Chuyển đề xuất từ Agent B/C thành thay đổi tọa độ có kiểm soát.
- Chỉ chỉnh những keypoint có căn cứ.
- Giữ nguyên các điểm đã tốt.
- Kiểm tra cấu trúc skeleton.
- Bảo đảm tọa độ theo độ phân giải ảnh gốc.
- Không ép mắt, miệng thành hình đối xứng khi mặt nghiêng.
- Không tự ý nội suy điểm không nhìn thấy trái guideline.
- Lưu bản sửa trước và sau dưới dạng JSON có thể rollback.

### Agent E — Independent QC

Đánh giá độc lập kết quả sau sửa. Không mặc định tin Agent B/C/D.

Phải xem ảnh gốc và annotation mới, phát hiện:

- Mắt lệch khỏi mí.
- Mày lệch khỏi cung mày.
- Môi lệch khỏi đường viền thật.
- Điểm nằm trên tay, kính hoặc nền.
- Keypoint trùng bất hợp lý.
- Cụm điểm bị co về giữa mặt.
- Sai trái/phải.
- Sai trạng thái outside/occluded.

Trả về `PASS`, `REWORK` hoặc `HUMAN_REVIEW`, kèm lý do cụ thể theo từng bộ phận.

---

## 4. FACE 1975 — ƯU TIÊN SỬA NGAY

Tập trung vào frame hiện tại: `G06_091.jpg`

Ảnh người dùng đang mở trên CVAT cho thấy các điểm mắt, mày và môi cần được kiểm tra bằng ảnh thực tế.

Frame này từng được Haiku đánh giá PASS, nhưng đánh giá PASS cũ không đủ chứng minh mọi tọa độ chính xác. Phải mở lại và kiểm tra độc lập.

Đồng thời xử lý các frame từng được báo cáo có lỗi:

- 95: môi trong bị dồn, khóe môi không hợp lý.
- 96: mắt, mày, môi sai khi mặt nghiêng.
- 97: môi lệch khỏi vùng mặt.
- 100: cấu trúc môi trong/ngoài bất thường.
- 104: cụm keypoint co về vùng nhỏ.
- 105: mắt/mày/môi dồn cụm.
- 107: bàn tay che vùng miệng.
- 112: IR tối, điểm dồn bất thường.
- 114: cụm mày và môi có điểm lệch.
- 102 và 106: kiểm tra nhãn còn thiếu.

Các frame còn lại cũng phải được kiểm tra, không tự động cho PASS theo báo cáo cũ.

**Mỗi frame phải có một kết luận QC độc lập dựa trên annotation mới nhất.**

---

## 5. CASE-BASED CORRECTION — KHÔNG SỬA ĐỒNG LOẠT

Các case khó cần nhận dạng: head rotation/yaw lớn; eyes closed/squinting; glasses occlusion; hand covering face; beard covering lips; low-light/infrared; small face/low resolution; mouth open/closed; partial face outside image; keypoints collapsed/geometry regression.

Với từng frame:

1. Nhận dạng case.
2. Tìm nguyên nhân sai.
3. Xác định những landmark nào đáng tin cậy.
4. So sánh tọa độ phiên bản cũ và mới.
5. Chọn bản tốt hơn có căn cứ, hoặc đề xuất sửa tiếp.
6. Tinh chỉnh cục bộ.
7. Render overlay mới.
8. Gọi Haiku độc lập kiểm tra.
9. Nếu chưa đạt, sửa tiếp vùng có vấn đề.
10. Nếu không đủ bằng chứng, chuyển human review.

Không để Haiku chỉ viết nhận xét rồi dừng mà không tạo thay đổi khi đã xác định được cách sửa hợp lệ.

---

## 6. QUẢN LÝ QUYỀN SỬA CVAT

Được phép sửa các keypoint đã xác định cần sửa thuộc job 1623 và 1975, trong phạm vi quyền hiện có, với điều kiện:

- Backup server trước khi ghi.
- Không xóa toàn bộ job.
- Không ghi đè các annotation đã tốt.
- Không thay thế skeleton bằng output thô của model.
- Chỉ patch điểm và thuộc tính cần thay đổi.
- Có log before/after.
- Có khả năng rollback.
- Không tự nhận là người dùng đã duyệt.
- Không thay đổi tên nguồn thành MANUAL để giả lập human review.
- Không submit bài hay push Git.

Phải kiểm tra version hoặc trạng thái server ngay trước khi ghi để tránh ghi đè chỉnh sửa mới của người dùng.

Sau mỗi batch: patch lên CVAT → đọc lại trực tiếp từ server → so sánh tọa độ và attributes với bản local → kiểm tra số skeleton và keypoint → render lại từ dữ liệu đã đọc trên server → chỉ ghi SYNC_VERIFIED khi dữ liệu khớp.

Nếu không được phép sửa bởi quyền truy cập thực tế, lưu patch local và báo BLOCKED. Không giả vờ đã sync.

---

## 7. HUMAN REVIEW — TỐI ĐA 5 ẢNH

Không gửi hàng chục ảnh để người dùng xem. Chỉ chọn tối đa 5 ảnh khó nhất còn bất đồng sau khi tự sửa.

Với mỗi ảnh: bản gốc, keypoint trước sửa, keypoint sau sửa, khoanh vùng nghi vấn, phóng lớn mắt/mày/môi, nêu rõ điểm và tọa độ cần quyết định, đưa hai phương án nếu cần.

Tạo: `reports/w2_recovery/HUMAN_REVIEW_TOP5.png` và mở trực tiếp trong VS Code.

Không tự coi đây là bước phê duyệt của người dùng.

---

## 8. QUALITY GATE — MỤC TIÊU 90/100

Rubric:

- 35 điểm: vị trí keypoint đúng theo ảnh và guideline.
- 20 điểm: chất lượng hình học mắt/mày/môi hoặc pose.
- 15 điểm: xử lý tốt occlusion, outside và case khó.
- 10 điểm: không làm regression so với bản tốt trước đó.
- 10 điểm: sửa đổi đồng bộ và read-back thành công.
- 10 điểm: evidence, audit trail và rollback.

Điểm phải dựa trên rubric và bằng chứng thực tế, không tự chấm 90 khi còn lỗi rõ.

PCK/NME chỉ tính khi có ground truth phù hợp. Không dùng đánh giá cảm tính của Haiku làm ground truth.

Nếu frame không đạt do che khuất, mờ hoặc guideline chưa rõ, chuyển trạng thái phù hợp, không sửa bừa để đạt điểm.

---

## 9. OUTPUT VÀ BẰNG CHỨNG

Tạo hoặc cập nhật trong `reports/w2_recovery/`:

- `RECOVERY_TIMELINE.md`
- `BEST_VERSION_COMPARISON.csv`
- `PER_FRAME_QC.csv`
- `REGRESSION_LOG.csv`
- `SYNC_VERIFICATION.csv`
- `HUMAN_REVIEW_TOP5.png`
- `FINAL_REPORT.md`

Cập nhật: `week02/issues.md`, `weeks/guidelines/w2-face50-pose17.md`.

Mỗi frame phải có: frame ID, danh sách case khó, phiên bản annotation được chọn, điểm được sửa, tọa độ trước và sau, lý do sửa, bằng chứng từ ảnh, kết quả QC, kết quả đồng bộ server, trạng thái cần người duyệt.

---

## 10. THỨ TỰ TRIỂN KHAI

- PHASE 1: Khôi phục lịch sử và bản annotation tốt nhất.
- PHASE 2: So sánh hiện trạng CVAT với các checkpoint.
- PHASE 3: Kiểm tra và sửa Face 1975, ưu tiên frame 91 và các case đã xác định có vấn đề.
- PHASE 4: Kiểm tra và sửa Pose 1623.
- PHASE 5: Independent QC, tự sửa các lỗi còn lại.
- PHASE 6: Đồng bộ CVAT và xác minh read-back.
- PHASE 7: Xuất tối đa 5 ảnh cần người duyệt và báo cáo cuối.

Các thao tác ghi lên cùng một frame phải được điều phối tuần tự. Không tạo lại framework nếu tools đã có. Không tải model mới nếu pipeline phiên sáng đã có. Không dừng ở báo cáo, inventory, review JSON hoặc chạy local inference.

**DEFINITION OF DONE**

- Đã truy vết được các phiên bản cũ có thể truy cập.
- Đã kiểm tra toàn bộ 60 frame.
- Đã sửa thực tế những keypoint có đủ bằng chứng.
- Không ghi đè kết quả tốt hơn bằng model output thô.
- Đã đồng bộ và read-back các thay đổi lên CVAT.
- Đã kiểm tra các case khó theo từng ảnh.
- Có tối đa 5 ảnh còn cần người duyệt.
- Có báo cáo rõ số điểm được sửa và chất lượng trước/sau.
- Mọi trạng thái PASS hoặc DONE đều có bằng chứng tương ứng.

**BẮT ĐẦU NGAY BẰNG HISTORY RECOVERY VÀ FRAME G06_091. KHÔNG CHẠY LẠI TOÀN BỘ MODEL KHI CHƯA XÁC MINH PHIÊN BẢN ANNOTATION TỐT NHẤT.**
