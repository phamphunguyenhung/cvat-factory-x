# Ghi chú tài liệu CVAT (job stage, state, review, xác thực)

Lấy ngày: 18/11/2026. Nguồn đọc qua web-fetch. Chỉ ghi những gì đã đọc được trên trang; phần chưa xác nhận được ghi rõ.

## 1. Trạng thái job (`state`) và giai đoạn (`stage`)

Nguồn: https://docs.cvat.ai/docs/getting_started/vocabulary/

- `state`: `new`, `in progress`, `rejected`, `completed`. Người được giao đổi trạng thái từ menu trong job.
- `stage`: `annotation`, `validation`, `acceptance`. Đặt từ dropdown ở trang task, ảnh hưởng thanh tiến độ của task.

## 2. Quy trình review (annotation → validation)

Nguồn: https://docs.cvat.ai/docs/qa-analytics/manual-qa/ (chỉ các bước giao diện)

- Chủ task (owner/maintainer) đặt Assignee là reviewer và Stage là `validation`.
- Để trả job về sửa: đặt Assignee là người sửa và Stage là `annotation`.
- Người sửa xử lý issue, rồi đổi state thành `completed`.
- Trang không nêu job chuyển sang stage nào sau khi reviewer chấp nhận.

Ý nghĩa với Task 473: "human review" theo CVAT là job ở stage `validation`. Tại thời điểm kiểm tra (18/11/2026), cả 6 job (1972–1976, 1623) đều ở `stage = annotation`.

## 3. Endpoint review

- Trang API có nhóm endpoint `reviews`, mô tả là "adding and removing the review of the job".
- **Chưa xác nhận:** đường dẫn cụ thể (ví dụ `/api/reviews?job_id=...`) và các giá trị `status` (accept / reject / review_further). Không tìm thấy trong các trang đã đọc.
- Trang Manual QA mô tả review là các issue gắn trên frame, không phải quyết định accept/reject.

## 4. Xác thực

Nguồn: https://docs.cvat.ai/docs/api_sdk/access_tokens/

- Header đúng: `Authorization: Bearer <token>`.
- Trang ghi cách xác thực bằng API key cũ (`Token <key>`) đã bị deprecated. Không có chi tiết định dạng cũ.
- Trong `lay_review.py` phải truyền `--auth-scheme Bearer`. Mặc định `Token` trả về 401.

## 5. Chưa lấy được

- Schema của `https://cvat.edufun.ai/api/docs/` chỉ trả về chuỗi "CVAT REST API". Swagger có thể nằm ở `<host>/api/swagger`, chưa thử.
- Nhóm endpoint trong https://docs.cvat.ai/docs/api_sdk/api/ chỉ có tổng quan, không có giá trị trường.

## 6. Nguồn đã đọc

- https://cvat.edufun.ai/api/docs/ (200, chỉ có tiêu đề)
- https://docs.cvat.ai/docs/api_sdk/api/
- https://docs.cvat.ai/docs/
- https://docs.cvat.ai/docs/api_sdk/access_tokens/
- https://docs.cvat.ai/docs/workspace/jobs-page/
- https://docs.cvat.ai/docs/qa-analytics/manual-qa/
- https://docs.cvat.ai/docs/getting_started/vocabulary/
