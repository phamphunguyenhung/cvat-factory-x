# Quy tắc làm bài

## Hình thức

- **Bài nhóm 5–6 người**, một repo mỗi nhóm.
- Được thảo luận khái niệm với nhóm khác, nhưng validator, review, bàn giao và truy vết do nhóm tự làm. Hai nhóm có validator giống nhau từng đoạn sẽ bị hỏi lại.
- **Mỗi người phải xem ảnh** ở Pha 2, và phải giải thích được phần mình làm khi vấn đáp.

## Validator

- Chỉ đọc thư mục `delivery/` được truyền vào. **Không đọc** `_private/`, `data_source/`, annotation COCO (kể cả file trong `data_source/cache/`), và **không dùng mạng**.
- Chỉ dùng thư viện chuẩn của Python. Được `import` `scripts/pngutil.py` (đọc PNG, thư viện chuẩn). Pillow chỉ được dùng cho công cụ xem ảnh của nhóm, không dùng trong validator.
- Không nhúng danh sách tên ảnh, id hoặc đáp án của bất kỳ seed nào vào code.

## Trung thực của số liệu

- Không bịa F1, kết quả review, hash, người gán/kiểm. Mọi số trong `README.txt` và `REPORT.md` phải lấy từ file thật.
- Không sửa `delivery/` (lô v1). Bản sửa là v2.
- Không sửa `scripts/` để qua bộ kiểm. Tìm thấy bug thật thì ghi vào `REPORT.md` và báo giảng viên.
- Không chạy `--dev-truth`, không đọc `_private/`, không dùng `score_findings.py --truth-dir` (chế độ của giảng viên). Giảng viên có thể hỏi nhóm đã chọn ngưỡng bằng cách nào.
- `score_findings.py` chỉ để **tự đo** validator tốt lên hay kém đi (in F1 làm tròn). **Dò đáp án** bằng cách nộp thử từng phần danh sách rồi xem điểm đổi thế nào cũng là đọc `_private` gián tiếp: bị coi là gian lận. `review_log.csv` phải là kết quả xem ảnh, không phải kết quả dò điểm.

## Dùng AI

- **Được** dùng AI để viết validator, giải thích khái niệm. **Bắt buộc** ghi prompt, chỗ AI sai và cách nhóm kiểm lại vào `submission/prompts.md`.
- **Không** dùng AI để thay việc xem ảnh. Mọi dòng `confirmed`/`rejected` là kết luận của người trong nhóm, ghi rõ ai xem.
- Ảnh COCO là dữ liệu công khai nên được dán vào công cụ AI để hỏi, nhưng phải ghi lại trong `prompts.md`. Ngoài đời, với dữ liệu thật của dự án thì **không** được làm vậy (xem `security.md`).

## Đề ẩn

- 40 phút, làm tại lớp trên máy của nhóm. Chỉ dùng validator và công cụ xem ảnh của nhóm.
- Không trao đổi kết quả với nhóm khác trong lúc làm.

## Git và nộp bài

- Mỗi người có ít nhất 2 commit có nội dung, mang tên và email của mình.
- Không commit `data_source/`, `runs/`, `*.zip`, secret. Bài làm chép sang `submission/` bằng `scripts/export_submission.py`.
- Bài chấm theo commit cuối trước hạn. Hạn và kênh nộp do giảng viên hoặc coach công bố.
