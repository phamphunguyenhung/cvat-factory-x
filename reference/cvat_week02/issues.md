# Issues — Week 2 / G06 (job 1623 pose17, job 1975 face50)

Cập nhật: 17/11/2026. Nguồn chuẩn: `HumanPose17_Annotation_Guideline_V1.docx` và `Face_Landmark50_Annotation_Guideline_V1.docx` (BTC). Có mâu thuẫn thì bản BTC thắng. Mọi quyết định của nhóm đánh dấu **Team clarification** hoặc **Experiment**, không gọi là quy định chính thức.

Trạng thái: `OPEN` chưa xử lý · `DECIDED` đã chọn hướng, đang làm · `NEEDS MENTOR` cần mentor chốt · `DONE` đã xong với bằng chứng.

---

## A. Lệch giữa guideline BTC và dữ liệu thật

| ID | Vấn đề | Nguồn | Quyết định của mình | Trạng thái |
|---|---|---|---|---|
| A1 | Pose: BTC ghi ảnh 960×540, job thực tế 1280×1024 | Pose §1 | Không đổi cách gán (BTC không nói kích thước ảnh ảnh hưởng nhãn). Ngưỡng ±5/±20 px giữ nguyên theo BTC. Hỏi mentor: ngưỡng pixel có cần scale theo ảnh thật không | NEEDS MENTOR |
| A2 | Face: BTC ghi ảnh 1280×720, job thực tế 640×480 | Face §1 | Như A1 | NEEDS MENTOR |
| A3 | Pose: BTC ghi "label person", CVAT job tạo label `person_vf17` | Pose §1 | Không đổi tên label (BTC cấm đổi tên và tạo label mới). Ghi lại để mentor xác nhận đây là cùng một label | NEEDS MENTOR |
| A4 | Face: BTC chưa nói quy tắc trái/phải khi mặt xoay ~90° trong ảnh fisheye. Pose thì BTC nói rõ kể cả ảnh xoay 90° | Face §1, Pose §1 | **Team clarification:** áp dụng "trái/phải theo vị trí trên ảnh" như câu chữ BTC cho face. Frame nào không phân định được thì UNCERTAIN_CLASS, không đoán | DECIDED (đang áp dụng, chờ mentor) |
| A5 | Face: BTC nói "Pre-label đã đúng: giữ vị trí nhưng vẫn đặt trạng thái cho cả 50 điểm" | Face §4 | Áp dụng nguyên văn: mọi điểm phải có trạng thái, dù pre-label đúng | DECIDED |
| A6 | Pose: BTC cho phép tinh chỉnh hoặc vẽ mới cho pre-label. Mười frame `source=file` là pre-label | Pose §1, Face §1 | **Đã làm theo yêu cầu của bạn (17/11):** đổi toàn bộ shape `source=file` sang `source=manual` bằng cách tạo lại, giữ nguyên tọa độ và trạng thái điểm. Job 1623: 10 shape. Job 1975: 70 shape (7 skeleton × 10 frame). Backup: `backups/before_manual_*.json` của từng job. Đã kiểm tra: tọa độ và nhãn giống hệt backup. **Lưu ý:** `source=manual` chỉ cho biết shape không còn từ file, không có nghĩa là đã được kiểm tra hay duyệt. Frame vẫn cần review và human approve. Vẫn cần mentor xác nhận câu hỏi 3 mục 8 | DONE (source đổi), chưa duyệt chất lượng |
| A8 | Theo yêu cầu (17/11): đổi toàn bộ nháp `source=auto` của tool sang `source=manual` trên CVAT, giữ nguyên tọa độ. Job 1623: 18 shape. Job 1975: 126 shape. Backup: `backups/before_manual_20261117_193049.json`. Đã kiểm tra: tọa độ giống backup, không còn `auto`. **Hệ quả:** id shape đã đổi, nên lệnh `pose rollback --run` dựa trên `run_*.json` không còn áp dụng cho các shape này. Muốn hoàn tác phải dùng backup. `source=manual` không có nghĩa là đã duyệt | DONE (source đổi), chưa duyệt chất lượng |
| A7 | Pose: BTC yêu cầu xét từng khớp, ví dụ tai bị tóc che hoàn toàn: occluded nếu ước lượng được, outside nếu không | Pose §2, §4 | Áp dụng nguyên văn | DECIDED |

## B. Dữ liệu và frame

| ID | Vấn đề | Bằng chứng | Quyết định | Trạng thái |
|---|---|---|---|---|
| B1 | Job 1975 thiếu ảnh local `G06_107.jpg` (frame 106) | `data/jobs/edufun/1975/frames` | Đã lấy lại từ CVAT bằng `Cvat.frame()` (chỉ đọc), 40 KB | DONE |
| B2 | Job 1623 frame 100 và 118 chưa có nhãn | `push_policy.json` ghi "vẽ tay" | Vẽ tay là việc bắt buộc theo BTC (mọi ảnh đều phải có skeleton). Đưa vào danh sách Haiku sửa tay, kiểm tra lại bằng overlay | OPEN |
| B3 | Job 1975 frame 102 chưa có nhãn | inventory `reports/w2_inventory_1975.csv` | Cùng cách như B2. Cần xem ảnh trước: nếu không có mặt lái thì bỏ qua theo BTC ("không có người lái thì bỏ qua") | OPEN |
| B4 | Face gold chỉ có 1 frame (f93) | `metrics/gold.json` | Không đủ để kết luận độ chính xác. Dùng Haiku review để có thêm bằng chứng định tính, không tuyên bố accuracy | OPEN |
| B5 | Pose chưa có gold | `eval_pose17.py`: "Chưa có gold" | Cần người sửa tay 5–8 frame. Không tự tạo gold từ nháp | NEEDS USER |

## C. Công cụ và môi trường

| ID | Vấn đề | Quyết định | Trạng thái |
|---|---|---|---|
| C1 | `eval_face50.py` lỗi `UnicodeEncodeError` trên console cp1252 | Chạy bằng `python -X utf8`. Sửa script để không phụ thuộc tham số | OPEN |
| C2 | `cvat_tools doctor` báo `ckpt False` cho `fan2d`, `vitpose_base`, `yolo11l_pose`, trong khi `models list` ghi smoke test đã chạy được (dạng HF/lazy load) | Chưa kết luận. Cần chạy inference thử một frame để xác nhận | OPEN |
| C3 | `weeks/2026-W47.yaml`: job 1975 ghi `kind: pose` (đúng schema) nhưng là face 50 điểm. Schema chỉ nhận `seg/det/pose/mot` | Giữ `pose`, ghi chú "face" trong comment. Không sửa schema khi chưa được phép | DECIDED |
| C4 | Trạng thái `local_status: todo` không khớp thực tế (đã có nháp trên CVAT) | Cập nhật khi có kết quả review | OPEN |
| C5 | File chưa theo dõi: `.pi/`, `scripts/export_tools_zip.py` | Chưa commit. Chờ người dùng quyết định | NEEDS USER |

## D. Quy trình và rủi ro

| ID | Vấn đề | Quyết định | Trạng thái |
|---|---|---|---|
| D1 | CVAT hiện chỉ đọc. Chưa có thao tác ghi nào trong phiên này | Không ghi lên server khi chưa có xác nhận rõ ràng. Mọi thay đổi chỉ nằm ở bản nháp local | DECIDED |
| D2 | Haiku review chỉ là đề xuất (VLM không đáng tin tuyệt đối về tọa độ pixel) | Đề xuất tọa độ của Haiku chỉ áp dụng khi overlay sau khi sửa được kiểm tra lại bằng mắt và có độ tin cậy vừa/cao. Không tự gọi kết quả là accuracy | DECIDED |
| D3 | Haiku không được tự approve | Trạng thái cuối luôn là `needs_human_review` nếu chưa có gold hoặc chưa được người duyệt | DECIDED |

---

## E. Kết quả review Haiku (bổ sung sau khi các agent hoàn tất)

### E1. Face 1975, frame 90–104 (đã xong)

Nguồn: `reports/w2_vlm/1975_90_104.json`. Tổng: pass 5, refine 0, uncertain 10, no_subject 0.

- **Pass:** 91, 93, 99, 101, 103.
- **Refine:** không có. Haiku không đề xuất tọa độ nào vì không frame nào đủ rõ, nên mục sửa tay của batch này trống.
- **Uncertain** (lý do ngắn):
  - 90: miệng nghi lệch 10–15 px, râu che môi, ảnh nhỏ.
  - 92: IR tối, mặt quay gần 90°, không xác nhận được mí nhắm và mép môi.
  - 94: cúi đầu, gọng kính che mắt, cụm mắt trái dẹt thành đường ngang.
  - 95: cúi đầu, cụm miệng xuống cằm, cụm mắt dẹt.
  - 96: mặt nhỏ nghiêng, mày và mắt lệch so với gọng kính, miệng lệch ra cằm.
  - 97: đầu ngửa nghiêng, mắt chồng lên gọng kính, miệng lệch.
  - 98: mặt quay mạnh, hai cụm mắt chồng gần giữa mặt, vi phạm quy tắc không dồn điểm.
  - 100: mặt quay nhỏ, hai cụm mắt chồng trong ~30 px, miệng lệch, râu che môi.
  - 102: không có nhãn trong `job1975.json`, overlay không có điểm nào. Mặt bị tay và kính che. Khớp với inventory (B3).
  - 104: kính râm che mắt, nghi Outside theo BTC. Toàn bộ 50 điểm dồn trong ~60×40 px, cụm miệng sát mép mặt.

**Lưu ý về độ tin cậy:** kích thước ảnh khác nhau (640×480, 960×540, 1280×720, 1280×1024, 512×424). Nhiều frame là IR tối. Đánh giá vị trí chỉ ở mức trung bình. Sai số ±5 px chưa kiểm chứng bằng số đo.

**Hệ quả:** 10/15 frame không đủ bằng chứng để Haiku đề xuất sửa. Các frame này cần người nhìn, hoặc cần một lượt review khác. Không tự áp dụng đề xuất nào từ batch này.

### E2. Face 1975, frame 105–119 (đã xong)

Nguồn: `reports/w2_vlm/1975_105_119.json`. Tổng: pass 7, refine 3, uncertain 5, no_subject 0.

- **Pass:** 108, 109, 113, 114, 115, 117, 119.
- **Refine:**
  - 105: mắt và mày bị dồn về giữa mặt, hai mắt cách nhau chỉ ~18 px, mặt nghiêng mạnh. Haiku không đề xuất tọa độ.
  - 107: miệng (30–49) nằm trên ngón tay, bàn tay che hoàn toàn. Đề xuất chuyển 20 điểm miệng sang outside (độ tin cậy vừa). Mắt trái và mặt dồn sát bàn tay, cần mentor xem.
  - 118: điểm 8 và 9 (đuôi mày phải) nằm trên nền cửa sổ, ngoài mặt. Đề xuất outside (độ tin cậy vừa). Cụm mắt do mặt nghiêng mạnh cần mentor xác nhận.
- **Uncertain:**
  - 106: chưa có nhãn, overlay không có điểm nào, ảnh IR tối. Khớp với inventory (chưa có nhãn, ảnh đã được lấy lại ở bước B1).
  - 110: ảnh IR rất tối, không xác nhận được mày, mắt, môi.
  - 111: đầu quay nghiêng mạnh, ảnh IR, không chắc ranh giới mắt.
  - 112: 50 điểm dồn trong vùng ~47×38 px, ảnh tối, không đề xuất tọa độ.
  - 116: mặt gần nghiêng 90° qua kính, ảnh IR, không chắc trục khuôn mặt.

**Lưu ý:** kích thước ảnh không đồng nhất (960×540, 1280×720, 1280×1024, 640×480, 512×424). Đánh giá dựa trên crop phóng 4×, không có đo lường tự động, nên các frame pass chỉ ở mức trung bình.

**Hệ quả:** đề xuất "outside" của 107 và 118 là gợi ý của Haiku, chưa áp dụng. Theo BTC, outside chỉ dùng khi không suy ra được vị trí, và nhóm chưa có quyết định cho trường hợp miệng bị tay che hoàn toàn. Cần mentor hoặc người nhìn trực tiếp trước khi sửa.

### E3. Pose 1623, frame 105–119 (đã xong)

Nguồn: `reports/w2_vlm/1623_105_119.json`. Tổng: pass 1, refine 6, uncertain 8, no_subject 0.

- **Pass:** 111 (cổ chân 16/17 outside, vì chân ra đáy khung; độ tin cậy vừa).
- **Refine:**
  - 107: gối phải (14) gán visible nhưng bị taplo che. Đề xuất occluded, giữ tọa độ (độ tin cậy vừa). Cổ chân 16/17 gán outside dù nằm trong khung.
  - 112: cổ tay phải (10) và trái (11) cùng visible và trùng tọa độ (cách ~8 px). Đề xuất 11 occluded, giữ tọa độ (độ tin cậy thấp).
  - 115: cổ chân 16/17 đặt trên viền cửa, cách gối 215 px, dài hơn đùi. Cần vẽ lại chân, không có đề xuất tọa độ.
  - 116: cánh tay phải (6, 8, 10) nằm trên mặt và cổ, ngoài thân. Cần vẽ lại, không có đề xuất tọa độ.
  - 118: chưa có skeleton nào. Khớp với inventory (B2). Cần vẽ mới.
  - 119: cổ chân trái (17) nằm trên gối trái (15), sai thứ tự. Đề xuất 17 → (672, 995) visible (độ tin cậy thấp). Gối phải (14) visible nhưng tối.
- **Uncertain:**
  - 105: ảnh xoay 90°, cặp R/L tách theo trục y nên không xác định chắc.
  - 106: ảnh đêm, người nằm ngang. Mọi cặp L có x lớn hơn R, có thể đảo nhưng chưa chắc.
  - 108: gối và cổ chân bị che, cặp R/L mâu thuẫn quy ước x.
  - 109: cổ chân phải nằm gần hông, không suy ra được vị trí thật.
  - 110: người nằm nghiêng xoay ngang, chân dưới và cặp hông, gối, cổ chân không chắc.
  - 113: cổ chân 16/17 nằm trên túi vàng. Cặp gối R/L ngược quy ước x.
  - 114: khuỷu phải (8) trùng vùng bàn tay với cổ tay (10). Cổ tay trái (11) nằm giữa cẳng tay.
  - 117: cổ chân 16/17 nằm trên túi giấy. Vai phải (6) sát cổ.

**Vấn đề cần mentor:** quy tắc R/L theo tọa độ x với người nằm ngang (105, 106, 110) chưa rõ. Guideline pose BTC nói R luôn phải ảnh, L luôn trái ảnh, kể cả ảnh xoay 90°, nhưng khi hai điểm cùng x thì không phân định được. Đã có mục này trong issues.md (A4, và câu 2 của guideline nhóm).

**Vấn đề cần kiểm tra:** nhiều cổ chân đang gán outside dù nằm trong khung (107, 111 là ví dụ). Cần kiểm tra lại có phải occluded không, vì BTC chỉ dùng outside khi ngoài khung hoặc không suy ra được vị trí.

### E4. Pose 1623, frame 90–104 (đã xong)

Nguồn: `reports/w2_vlm/1623_90_104.json`. Tổng: pass 1, refine 9, uncertain 5, no_subject 0.

- **Pass:** 90.
- **Refine:**
  - 91: tai phải visible nhưng bị tóc che. Tai trái trùng mắt trái (3/5). Vai trái lệch vào thân. Hai hông (12, 13) nằm trên cẳng tay.
  - 92: mũi (1) nằm dưới cằm, lệch ~35 px. Gối trái (15) thấp hơn cổ chân trái (17). Khuỷu phải (8) nằm ngoài người. Khớp với ảnh chụp màn hình của bạn: chân phải đang gán nét đứt, dù nằm trong khung.
  - 93: vai trái (7) thấp hơn khớp vai thực ~70–90 px. Cổ chân outside dù nằm dưới táp-lô, cần ATTRIBUTE_CHECK.
  - 94: cổ chân trái (17) đặt trên tay cầm báo và gán outside. Vai trái (7) lệch vào trong ngực.
  - 95: cổ chân phải (16) nằm trên đùi, gán outside sai. Cổ chân trái (17) outside khi chân còn thấy tới đáy khung.
  - 98: cổ chân phải (16) và trái (17) đặt trên cửa phải, không phải vùng chân.
  - 99: gối trái (15) và cổ chân trái (17) gán outside khi chân còn trong khung. Vai trái (7) lệch vào ngực.
  - 100: không có nhãn. Khớp với inventory (B2). Cần vẽ mới, không có đề xuất tọa độ vì ảnh mờ do chuyển động.
  - 101: mắt phải (2) và tai phải (4) trùng tọa độ (265, 141) nhưng khác trạng thái.
- **Uncertain:**
  - 96: ảnh xoay 90°, R/L không tách được theo trục x. Gối và cổ chân bị che.
  - 97: gối và cổ chân nằm trên ghế phụ và cửa, không suy ra được chuỗi chân.
  - 102: gối phải và cổ chân phải nằm trong vùng tối dưới vô lăng. Khớp với inventory (B3 cho job 1975, và đây là job 1623).
  - 103: ảnh tối và xoay. Mắt và tai phải gần trùng (~4 px). Hông và gối có khả năng đảo R/L.
  - 104: mắt và tai phải trùng (~2 px). Có thể có cánh tay người thứ hai ở phía phải ảnh (UNCERTAIN_SCOPE).

**Hệ quả:**
- Lỗi lặp lại nhiều nhất: cổ chân gán outside khi vẫn nằm trong khung. Lỗi này xuất hiện ở 92, 93, 94, 95, 98, 99, 107, 111 và 113. Cần sửa thành occluded hoặc giữ visible theo bằng chứng, không nên gán outside mặc định.
- Vai trái (7) bị lệch vào thân ở nhiều frame (91, 94, 99). Đây là lỗi hệ thống của nháp, không phải lỗi từng frame.
- Frame 100 không có trong `w2_current`. Điều này khớp với inventory, nên không phải lỗi xuất dữ liệu. Frame 100 thật sự chưa có nhãn.

Đã xong cả bốn batch Haiku (pose 1623 và face 1975, mỗi job 30 frame). Chưa có đề xuất nào được áp dụng.

## F. QC cuối (17/11/2026)

Báo cáo đầy đủ: `reports/W2_G06_FINAL_QC_REPORT.md`. Trạng thái: `PARTIAL_COMPLETE`, không có frame nào `human_approved`.

| ID | Vấn đề | Trạng thái | Ghi chú |
|---|---|---|---|
| F1 | Pose f101, f104: mắt phải (2) và tai phải (4) trùng tọa độ | NEEDS HUMAN DECISION | REVIEW_02. Chưa sửa: không biết điểm nào đúng |
| F2 | Pose f100, f118 và face f102, f106 chưa có nhãn | OPEN | Cần vẽ mới. Nếu không có người lái thì bỏ qua theo BTC |
| F3 | Pose R/L với người nằm ngang hoặc ảnh xoay (15 frame uncertain, ví dụ f110) | NEEDS MENTOR | REVIEW_01. Rule trục x không kiểm được cho các frame này |
| F4 | Pose f94, f99: gối và cổ chân bị che nhưng gán outside | NEEDS FIX (Haiku đề xuất occluded) | Chưa áp dụng: cần người xác nhận |
| F5 | Face 9 frame cần sửa (95, 96, 97, 100, 104, 105, 107, 112, 114), có dồn điểm và môi lệch | OPEN | Cần vẽ lại. Đề xuất tọa độ của Haiku không đủ tin cậy |
| F6 | Face f107: môi bị tay che nhưng vẫn visible | NEEDS HUMAN DECISION | REVIEW_03 |
| F7 | Face f112: 50 điểm dồn trong ~47×38 px, IR tối | NEEDS HUMAN DECISION | REVIEW_04 |
| F8 | Face f92, f96: khó xác định hướng mặt và cụm môi | UNCERTAIN | Có trong `human_review_queue.csv`, chưa mở ảnh riêng vì đã đủ 4 ảnh |
| F9 | Chưa đo được sai số ±5 px: pose không có gold, face chỉ 1 frame | NOT_MEASURED | Cần bạn sửa tay 5–8 frame pose để có gold |
| F10 | Đổi nguồn `auto`/`file` sang `manual` làm id shape thay đổi, rollback theo run không còn dùng được | DONE, đã ghi trong A8 | Dùng backup để hoàn tác |
