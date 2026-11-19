# lay-review-cvat

**Giải quyết:** cần số liệu tiến độ và issue của job CVAT để viết nhật ký tuần, không phải đếm tay trên CVAT.

## Tool làm gì

Đọc qua CVAT REST API các job bạn chỉ định, rồi in ra:

- trạng thái job (`state`, `stage`), người được giao, khung frame
- số CVAT issue (mở / tổng) và nội dung bình luận đầu tiên
- số shape theo nguồn (`manual`, `auto`, ...) và số frame đã có shape

Kết quả ghi ra `du-lieu-review/review_cvat.json` và `review_cvat.md` ở thư mục gốc repo. Thư mục này nằm trong `.gitignore`, không commit.

Mặc định chỉ đọc. Tham số `--apply` là ngoại lệ duy nhất: nếu job đang `new` và đã có shape thì chuyển sang `in progress`. Tool không bao giờ đặt `completed`, vì người gán tự bấm hoàn thành.

## Chuẩn bị

1. Lấy token CVAT (PAT) của chính bạn trong CVAT, mục cài đặt tài khoản. Không dùng token của người khác.
2. Tạo file `.env.local` ở gốc repo, một dòng:
   ```
   CVAT_TOKEN=<token của bạn>
   ```
   File này đã nằm trong `.gitignore`. Không dán token vào chat hay vào file được commit.
3. Cài thư viện nếu chưa có:
   ```
   pip install requests
   ```

Script không tự đọc `.env.local`. Bạn phải nạp biến trước khi chạy.

**Bash / Git Bash:**
```bash
set -a; . ./.env.local; set +a
```

**PowerShell:**
```powershell
Get-Content .env.local | ForEach-Object { $k, $v = $_ -split '=', 2; [Environment]::SetEnvironmentVariable($k, $v) }
```

## Cách chạy

Chạy từ gốc repo. Luôn thêm `--auth-scheme Bearer`, vì mặc định của script là `Token` và sẽ bị 401.

```bash
# Lấy tự động mọi job của Task 473 và Task 402
python -X utf8 source-tool/lay-review-cvat/lay_review.py --task 473,402 --auth-scheme Bearer

# Xem trạng thái job 1972 đến 1976 của Task 473
python -X utf8 source-tool/lay-review-cvat/lay_review.py --jobs 1972-1976 --auth-scheme Bearer

# Xem job 1623 của Task 402
python -X utf8 source-tool/lay-review-cvat/lay_review.py --jobs 1623 --auth-scheme Bearer

# Chạy và chuyển job đang new sang in progress nếu đã có shape
python -X utf8 source-tool/lay-review-cvat/lay_review.py --jobs 1972-1976 --auth-scheme Bearer --apply
```

Cần truyền `--task` hoặc `--jobs`. Không truyền gì thì tool báo lỗi và dừng.

Nên chạy lần đầu không có `--apply` để xem đề xuất. Chỉ thêm `--apply` khi đã đồng ý với đề xuất.

## Tham số

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--task` | — | Danh sách task, ví dụ `473,402`. Tool tự lấy mọi job của task qua API |
| `--jobs` | — | Danh sách job, ví dụ `1972-1976,1623`. Dùng khi chỉ muốn một số job cụ thể |
| `--url` | `https://cvat.edufun.ai` | Địa chỉ CVAT |
| `--org` | `cohort-5a` | Tổ chức CVAT |
| `--auth-scheme` | `Bearer` | Giữ mặc định. Không đổi thành `Token` |
| `--apply` | tắt | Ghi `new → in progress` cho job đã có shape |
| `--out` | `du-lieu-review/` ở gốc repo | Thư mục ghi kết quả |

## Đầu ra

- `review_cvat.md`: bảng job, frame, người giao, stage, state, số issue mở/tổng, số shape auto, số frame còn shape auto. Phía dưới là danh sách issue, có link tới đúng frame.
- `review_cvat.json`: dữ liệu thô đầy đủ, dùng khi cần đối chiếu.

Khi một job không đọc được, script ghi lỗi kèm mã HTTP và chạy tiếp các job khác.

## Lưu ý

- **HTTP 401:** token sai hoặc thiếu `--auth-scheme Bearer`.
- **HTTP 403 trên job:** token không có quyền trên task đó. Ví dụ token worker không đọc được Task 402, chỉ đọc được job 1623. Cần người có quyền Owner hoặc Maintainer trên task.
- **Số shape khác báo cáo cũ:** lấy số trong `review_cvat.json` làm chuẩn, rồi ghi cả hai số vào nhật ký nếu đang đối chiếu.
- **Không đo chất lượng.** Số frame có shape chỉ đo tiến độ gán. Chất lượng kiểm tra bằng mắt ở bước review.
- Đầu ra mang tên người và MSSV của CVAT. Đừng đăng nguyên file lên nơi công khai.

## Người viết

Bùi Văn Thịnh (tuần 1). Tuần 2: đổi mặc định sang `Bearer`, thêm `--apply` chuyển `new → in progress`.
