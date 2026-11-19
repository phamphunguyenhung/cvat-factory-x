# review-assign — chia reviewer chéo

**Giải quyết:** Mỗi lần có task/job mới phải tự nghĩ ai review ai, dễ quên và dễ để người gán tự review job của mình.

## Tool làm gì

Nhận danh sách thành viên và map `job → annotator`, trả về reviewer cho từng job theo 2 quy tắc:

- **Cứng:** reviewer không bao giờ là annotator của job đó. Ai trong `exclude_reviewers` (ví dụ Lead) không nhận việc review.
- **Mềm:** cân bằng số job review; khi hoà thì xoay vòng theo thứ tự thành viên.

Tool **chỉ đọc đầu vào, không gọi CVAT, không ghi gì lên server**. CVAT không có trường "reviewer" cho job, nên kết quả được ghi vào nhật ký tuần.

## Cách chạy

```powershell
cd D:\06-Learning\AIA-VINGROUP\a48\source-tool\review-assign
python -X utf8 assign_reviewers.py tuan-02-task473.json
python -X utf8 assign_reviewers.py tuan-02-task473.json --out reviewers.json
python -m pytest -q .
```

Khi có task mới: copy `tuan-02-task473.json`, sửa `members` và `jobs`, chạy lại. Dán bảng in ra vào mục **Công việc** của nhật ký task tương ứng: `nhat-ky-tuan/tuan-NN/task-<số>-<loại>_<lead>.md`.

## Đầu vào

```json
{
  "members": ["A", "B", "C"],
  "exclude_reviewers": ["Lead"],
  "jobs": [{"id": 1972, "annotator": "A"}]
}
```

Tên trong `jobs[].annotator` phải khớp đúng từng chữ với `members`. Tool báo lỗi nếu không khớp.

## Giới hạn

- Chỉ chia được khi có ít nhất 2 người không bị loại. Nếu chỉ còn 1 người hợp lệ, tool báo lỗi thay vì tự đoán.
- Không kiểm tra khối lượng công việc theo số frame. Mỗi job tính như nhau.
- Quyết định có loại Lead khỏi việc review hay không là của đội, không phải của tool. Mặc định tool không loại ai.

## Người viết

Phạm Phú Nguyễn Hưng

## Đẩy lên CVAT (`apply_cvat.py`)

Chia reviewer rồi đặt `stage=validation` + `assignee=reviewer` cho các job đã xong. Chỉ job nêu trong `--jobs` bị ghi; mặc định dry-run, `--yes` mới ghi thật, có backup + `--rollback`.

```powershell
python -X utf8 apply_cvat.py tuan-02-task473.json --task 473 --jobs 1975 1976 --yes
```
