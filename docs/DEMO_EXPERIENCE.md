# Trải nghiệm demo — "Job CVAT mới tinh" (CVAT Factory X)

Nguyên tắc: demo diễn ra trong một **task/job CVAT tạo mới hoàn toàn trên CVAT LOCAL** (`http://localhost:8080`, docker),
ảnh **chưa có annotation**. Không chạm `cvat.edufun.ai`, không chạm job của thành viên khác.
Ảnh lấy từ dataset thật của bài lab Day 10 (`demo\coco128_clean`, gốc COCO val2017, CC BY 4.0). Scene/job/người gán nhãn của lab là **giả lập**.

## Luồng người xem thấy (5 phút)

| # | Màn hình | Điều xảy ra | Ai làm |
|---|---|---|---|
| 0 | Dashboard → "Agent Console" | Agent chạy `ingest`: tạo task `RC-DEMO-<giờ>` trên CVAT local, upload N ảnh (mặc định 20), 1 job, nhãn từ taxonomy. Hiện link mở job. Trong CVAT: **job trống, 0 shape**. | Agent (CVAT API, chỉ task do chính nó tạo) |
| 1 | Agent Console | `prelabel`: YOLO sinh nhãn nháp ghi vào đúng job đó (`source=auto`, kèm model version + confidence). Mở CVAT thấy box nháp. | Agent |
| 2 | Agent Console | `route`: chấm điểm rủi ro từng frame (không phải bằng chứng đúng/sai), xếp hàng review, frame rủi ro cao đứng đầu. | Agent |
| 3 | CVAT job (tab thật) | Người duyệt **sửa tay vài frame**. Sửa xong bấm "Pull". Phê duyệt QC/privacy và consumer ACK **chỉ có khi người thật truyền `--approver` / `--consumer`**. | Con người |
| 4 | Agent Console | `pull`: export annotation của job → dataset + manifest + sha256 → snapshot v1 → `validate` → VALIDATED. | Agent |
| 5 | Agent Console | `inject` T6+T5 lên **bản sao local** (không đẩy lỗi lên CVAT) → vẫn parse được → `validate` → HOLD + bằng chứng. | Agent |
| 6 | Lineage / Audit | Graph chỉ cạnh có bằng chứng; cạnh giả lập gắn `simulated`. Audit log hiện từng hành động của agent. | — |
| 7 | Version | `restore` snapshot sạch → so digest → validate lại → release v2 (PUBLISHED chỉ khi có approval thật; ACCEPTED chỉ khi có ACK thật). | Agent + người |

## Agent giao tiếp thế nào
- Agent **không phải LLM tự quyết**: là bộ lệnh cố định (`ingest, prelabel, route, pull, validate, inject, trace, restore`) gọi từ Agent Console (dashboard) hoặc CLI. LLM/VLM chỉ là tư vấn tùy chọn, không phải oracle release.
- Mỗi lệnh: kiểm tra **write-guard** → thực thi → ghi **audit log** (hash-chained JSONL) → trả kết quả có cấu trúc kèm link CVAT.
- Write-guard: chỉ cho ghi vào CVAT khi (a) host là localhost, (b) task id nằm trong `session.json` do chính agent tạo, (c) `RC_READ_ONLY` ≠ 1 cho thao tác ghi. Vi phạm → từ chối + audit `DENIED`.

## Hợp đồng liên module (các agent code theo đúng tên này)
- `artifacts/release_control_demo/session.json`: `{cvat_host, task_id, job_id, task_name, image_files[], label_map{}, created_at, created_by:"release_control"}`
- `release_control.demo_session.session`: `create_session(n_images=20)->dict`, `load_session()->dict`, `cleanup_session()` (chỉ xóa task đã ghi trong session.json), class `CvatDemoClient` (`get_job_annotations()`, `put_annotations(shapes)` — bị write-guard).
- `release_control.audit`: `AuditLog(path).append(actor, action, target, inputs=None, outputs=None, evidence=None)->dict`, `.read(limit=None)->list`, `.verify_chain()->bool`; `guard_write(host, task_id, session)` raise `WriteDenied`.
- `release_control.agent.prelabel.run_prelabel(session)->dict`; `release_control.agent.routing.route(session)->dict` (dùng `release_control.scale.risk/review_queue`).
- `release_control.agent.pull.pull_job_to_dataset(session, out_dir)->Path` (ra thư mục có `manifest.json` đúng định dạng `validators/structural.py`).
- CLI mới `python -X utf8 -m release_control.cli_live <ingest|prelabel|route|pull|validate|inject|trace|restore|live|cleanup>`; `live` chạy tuần tự các bước tự động, dừng ở bước 3 chờ người.

## Thang chấp nhận của trải nghiệm
Job CVAT mới có 0 shape → sau `prelabel` có shape `source=auto` → audit log có đủ dòng và `verify_chain()` đúng → HOLD khi inject → restore khớp digest → không có approval/ACK/training bịa.
