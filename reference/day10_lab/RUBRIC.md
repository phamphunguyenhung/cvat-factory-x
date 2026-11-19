# Rubric — 100 điểm

Điểm dựa trên **kết quả đo được** (đề ẩn), **bằng chứng trong repo** và **vấn đáp**, không dựa vào việc lô có Release hay không. Mỗi mục chấm theo bậc: **đủ điểm** khi đạt hết tiêu chí, **một nửa** khi đạt phần lớn nhưng có lỗi không nghiêm trọng, **0** khi thiếu hoặc sai căn bản.

**Vấn đáp (bắt buộc, khoảng 3 phút/người, người chấm chọn ngẫu nhiên):**
1. Giải thích một đoạn validator và sửa nó khi đổi yêu cầu.
2. Giải thích một dòng `confirmed` hoặc `rejected` trong `review_log.csv`: đã thấy gì trong ảnh.
3. Giải thích một con số trong `README.txt` của lô v2.

Người nào không trả lời được thì phần điểm cá nhân (mục 6) tối đa một nửa.

| # | Hạng mục | Điểm | Bằng chứng chính |
|---|---|---:|---|
| 1 | Bắt lỗi trên đề ẩn | 35 | `de_an/errors.json`, chấm bằng `score_findings.py` |
| 2 | Review bằng mắt | 15 | `de_an/review_log.csv` (chấm độ đúng); `submission/r<seed>/review_log.csv` (chỉ chấm cách ghi) |
| 3 | Bàn giao v2 trên seed luyện tập | 25 | `submission/r<seed>/handover/`, `check_handover.py` |
| 4 | Truy vết | 10 | `de_an/trace.json`, `submission/r<seed>/trace.json` |
| 5 | Bảo mật | 5 | `security.md` |
| 6 | Quy trình, trung thực, vấn đáp | 10 | lịch sử commit, `prompts.md`, `REPORT.md`, vấn đáp |

## 1. Bắt lỗi trên đề ẩn (35)
Trước khi chấm, mọi lỗi gán nhãn trong `de_an/errors.json` **không có dòng `confirmed`** (ghi chú ≥ 20 ký tự) trong `de_an/review_log.csv` bị **bỏ ra**. Tự nhận hết ứng viên mà không xem ảnh chỉ được điểm như chỉ kiểm luật.
- **20 — F1 tuyệt đối.** Điểm = 20 × `rank F1` (làm tròn 0,5). F1 thấp hơn 0,5 hoặc không có thì 0.
- **10 — Xếp hạng giữa các nhóm** theo `rank F1`. Nhóm đứng đầu được 10, nhóm cuối được 2, các nhóm giữa nội suy.
- **5 — Lỗi gán nhãn.** F1 của nhóm `cross_check_review` ≥ 0,6 được đủ điểm; từ 0,3 đến dưới 0,6 được một nửa.
- *Trừ nặng:* validator đọc `data_source/`, annotation COCO, `_private/`, hoặc dùng mạng: **0 điểm mục 1** và báo giảng viên.

## 2. Review bằng mắt (15)
- **8 — Đề ẩn: mọi lỗi gán nhãn trong `errors.json` có dòng `confirmed`.** Ghi chú nói **thấy gì** trong ảnh. Ghi "có lỗi" hay "OK" thì chỉ được một nửa. Ở seed luyện tập, nhóm có công cụ tự đo nên chỉ chấm cách ghi, không chấm độ đúng.
- **4 — Phân biệt lỗi với lệch tự nhiên.** Có dòng `rejected` kèm lý do (vật bị che, crowd); nếu hàng đợi không có ca lệch tự nhiên thì `REPORT.md` cho thấy nhóm đã đo trên lô sạch và giải thích vì sao ngưỡng không gây báo thừa.
- **3 — Đề ẩn: phân biệt đúng `class_swap` / `category_swap`** dựa vào ảnh, không đoán.

## 3. Bàn giao v2 (25)
- **10 — `check_handover.py` không còn FAIL** (H1–H17, chạy lại bằng bản gốc của script).
- **6 — `rework_queue.csv` và `README.txt`.** Ảnh lỗi giao đúng người; kết luận v1 và v2 kèm mẫu số, đúng ngưỡng 2%; phần Ghi chú nói rõ đã sửa gì trong v2 (hash, guideline, split) và vì sao.
- **5 — Lineage.** HOLD cho v1, release v2 có parent; `REPORT.md` giải thích vì sao `R-021` phải xem lại hoặc train lại.
- **4 — Không ghi đè.** v1 giữ nguyên, nhãn trong v2 không sửa tay; `REPORT.md` giải thích hash chứng minh điều gì và **không** chứng minh điều gì.

## 4. Truy vết (10)
- **6 — Đề ẩn:** trung bình F1 của `tier_a`, `tier_b`, `notify`, nhân 6.
- **4 — Seed luyện tập:** `trace.json` đúng, và `REPORT.md` nêu tiêu chí đã dùng (guideline theo job, không theo ảnh).

## 5. Bảo mật (5)
- `security.md` đủ ba mục, gắn với lô vừa làm. Ca khuôn mặt kết luận **Hold + báo người phụ trách**, nêu ví dụ ảnh có người trong lô. Chép slide thì được 0.

## 6. Quy trình, trung thực, vấn đáp (10)
- **4 — Commit.** Mỗi người có ít nhất 2 commit có nội dung; thứ tự hợp lý theo pha.
- **3 — Dùng AI minh bạch.** `prompts.md` ghi prompt, chỗ AI sai và cách nhóm kiểm lại.
- **3 — `REPORT.md`.** Trả lời câu "vì sao báo hết ứng viên làm F1 giảm", nêu số liệu thật của nhóm (F1 luyện tập theo từng nhóm lỗi, số ứng viên trên lô sạch).

## Trừ điểm chung
- **Số liệu giả** (F1, kết quả review, hash bịa): 0 điểm phần liên quan.
- **Sửa script trong `scripts/`** để qua bộ kiểm: 0 điểm mục 3. Người chấm chạy bản gốc.
- Nộp muộn, sai tên repo: theo [RULES.md](RULES.md).
