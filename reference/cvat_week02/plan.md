# MISSION: CVAT AUTOLABEL FACTORY V3 — UNIVERSAL CLI & MODEL-DRIVEN ANNOTATION

## VAI TRÒ

Bạn là Principal Python Engineer, Computer Vision Engineer và CVAT SDK Architect.

Nhiệm vụ: biến toàn bộ công cụ đã phát triển trong repository CVAT hiện tại thành một sản phẩm CLI thống nhất, có thể cài đặt và tái sử dụng cho nhiều CVAT server, task, job và annotation type.

**EXECUTE, DON'T JUST PLAN.**

Tự nghiên cứu codebase, xác minh, refactor, lập trình, kiểm thử và chạy thử. Không dừng ở việc viết tài liệu hoặc kiến trúc đề xuất.

Ưu tiên tái sử dụng code đã hoạt động, không xây lại từ đầu.

## 1. SOURCE OF TRUTH — KHÔI PHỤC HỆ THỐNG HIỆN TẠI

Repository dự kiến:

`D:\06-Learning\AIA-VINGROUP\learning\cvat`

Workspace người dùng đang mở có thể là `learning\cvat\week02`.

Trước tiên xác nhận Git root và code root thực sự. Không giả định thư mục `week02` chứa toàn bộ source.

Đọc và kiểm kê:

- `rules.md`
- `src/cvat_tools/`
- `src/cvat_tools/autolabel/`
- `src/cvat_tools/pose/`
- `tests/`
- `pyproject.toml` và các dependency files
- `docs/guidelines/`
- `data/autolabel_cache/`
- `artifacts/`
- `~/.gemini/config/skills/cvat-image-first-sync/SKILL.md`

Đặc biệt tìm:

- `face50.py`
- `face50_engine.py`
- `cache.py`
- `qc.py`
- `refine.py`
- `sync.py`
- `cli.py`
- `__main__.py`
- Các engine Pose17 đã có
- Code segmentation, BBox, tracking, SAM, YOLO đã từng phát triển

Đọc Git history và trạng thái working tree để tìm những implementation hữu ích.

Không reset, checkout, clean, force-push hoặc xóa file đang sửa.

**QUAN TRỌNG: người dùng đã chỉnh thủ công nhiều annotation sau phiên trước. Live CVAT hiện tại phải được xem là trạng thái mới nhất, không được ghi đè bằng proposal cũ.**

Các job tham chiếu:

- Pose17: Job 1623.
- Face50: Job 1975, Task 473.
- Server: `https://cvat.edufun.ai`

Chỉ dùng hai job này để kiểm tra khả năng tái sử dụng. Không tự sửa annotation đang có.

## 2. TARGET PRODUCT

Tên dự án:

**CVAT Autolabel Factory**

Tên CLI mong muốn:

`cvat-factory`

CLI cần hỗ trợ:

```bash
cvat-factory doctor
cvat-factory setup
cvat-factory discover
cvat-factory models list
cvat-factory jobs
cvat-factory inspect --job 1975
cvat-factory run --job 1975 --mode auto
cvat-factory review --job 1975
cvat-factory sync --job 1975
cvat-factory benchmark --job 1975
cvat-factory resume --job 1975
```

Đây là giao diện đích. Phải triển khai command thật, không để placeholder. Nếu không thể hỗ trợ một command, ghi rõ trạng thái và không tuyên bố hoàn thành.

Sử dụng Python package có console entrypoint chuẩn trong `pyproject.toml`.

Không fork CVAT CLI khi không cần. Xây extension dựa trên CVAT SDK/CLI chính thức.

Đọc:

- https://docs.cvat.ai/docs/api_sdk/cli/
- https://docs.cvat.ai/docs/api_sdk/sdk/
- https://docs.cvat.ai/docs/api_sdk/access_tokens/

Kiểm tra phiên bản SDK tương thích với server thực tế. Không nâng cấp dependency mù quáng.

## 3. ZERO-FRICTION SETUP

### 3.1. Credential discovery

Khi chạy lần đầu:

1. Tìm CVAT profile có sẵn.
2. Kiểm tra cấu hình môi trường.
3. Kiểm tra Windows Credential Manager hoặc secure credential store được hỗ trợ.
4. Kiểm tra khả năng kết nối server.
5. Xác thực token hiện tại.
6. Nếu thiếu token, hướng dẫn người dùng nhập PAT qua prompt bảo mật.
7. Lưu credential qua cơ chế an toàn, không lưu plaintext vào repository.

Không được in PAT ra terminal, log, báo cáo hoặc agent context.

Không tìm cách lấy token từ browser session, cookie hay Chrome profile.

Không tự ý cấp quyền hoặc giả lập quyền.

### 3.2. Role detection

Tự nhận diện:

- Identity đang đăng nhập.
- Organization.
- Project/task membership.
- Job assignee.
- Owner/maintainer/supervisor/worker nếu API hỗ trợ.
- Quyền đọc, tạo, cập nhật annotation thực tế.

Không suy ra quyền ghi chỉ từ tên role.

Kiểm tra permission theo chức năng API thực sự.

Nếu chỉ được đọc, tự chuyển sang read-only mode.

### 3.3. Task/job discovery

Tự động đọc danh sách job có thể truy cập.

Hiển thị:

- Job ID.
- Task ID.
- Task name.
- Job status.
- Assignee.
- Số frame.
- Annotation type.
- Danh sách label.
- Annotation coverage.
- Model phù hợp.
- Trạng thái xử lý gần nhất.

Nếu chạy không chỉ định job, ưu tiên resume job dở dang từ lịch sử local. Nếu có nhiều job, cung cấp menu chọn ngắn gọn; không tự ghi vào một job được phỏng đoán.

## 4. MODEL REGISTRY — TÌM LẠI TOÀN BỘ MODEL CŨ

Đây là bước bắt buộc trước khi tải thêm model.

Tìm model đã có trong các vị trí liên quan:

- Repository hiện tại.
- Local model cache.
- Hugging Face cache.
- PyTorch cache.
- Ultralytics cache.
- ONNX models.
- OpenVINO models.
- Các thư mục model được cấu hình.
- WSL model locations nếu có quyền.
- Existing Python virtual environments.

Quét có giới hạn theo đường dẫn và độ sâu. Không quét vô hạn toàn bộ ổ đĩa.

Nhận diện:

- Model name.
- Architecture.
- Weight path.
- Version.
- SHA256.
- Checkpoint size.
- Framework.
- Supported task.
- Input size.
- CPU/GPU compatibility.
- VRAM/RAM requirements.
- Expected label schema.
- License/source nếu xác minh được.
- Trạng thái đã test và benchmark.

Tìm các model và implementation từng được sử dụng, gồm:

- FAN2D FaceLandmarker.
- Face50 custom alignment.
- Pose17 model.
- YOLO detection/segmentation.
- SegFormer.
- SAM/SAM2.
- Các model tracking.
- Mọi model khác thực tế đang có.

Không kết luận model đã tồn tại chỉ dựa trên lịch sử chat.

Không tải lại model trùng hash.

Tạo SQLite registry có version và provenance.

Cho phép benchmark nhiều model trên cùng một bộ ảnh mẫu, so sánh chất lượng, latency và chi phí tính toán.

Không chọn model chỉ dựa trên tốc độ hoặc confidence.

## 5. UNIVERSAL ANNOTATION ARCHITECTURE

Thiết kế plugin/adapter architecture.

Hỗ trợ bốn nhóm annotation chính:

### A. Detection — BBox

- Nhận diện taxonomy.
- Prelabel bằng model phù hợp.
- Map label ID đúng CVAT task.
- Kiểm tra bbox geometry.
- Duplicate/overlap detection.
- Missing-object candidate detection.
- Small-object handling.
- Selective image tiling.
- Confidence calibration.
- Batch inference.

Tận dụng guideline 7 lớp giao thông nếu tìm thấy trong repository.

### B. Segmentation

- Semantic segmentation.
- Instance segmentation.
- Mask.
- Polygon.
- Boundary refinement.
- Shared-boundary constraints.
- Layer ordering.
- Polygon simplification.
- Self-intersection validation.
- Mask read-back.

Tìm và tái sử dụng Shape-first annotation PoC nếu có.

Ưu tiên tạo vùng lớn bằng geometric primitives, refine boundary theo ảnh thật, chuyển đổi mask/polygon theo định dạng cần thiết.

### C. Tracking

- Object identity across frames.
- Temporal consistency.
- Keyframe detection.
- Interpolation.
- Occlusion handling.
- Track fragmentation detection.
- ID-switch detection.
- Frame-to-frame validation.

Không tự tạo hoặc đổi track ID nếu chưa có bằng chứng liên kết đủ mạnh.

### D. Skeleton / Keypoints

Hỗ trợ:

- Pose17.
- Face50.
- Custom skeleton schema.

Không hardcode landmark count vào universal engine.

Bắt buộc giữ:

- Keypoint ordering.
- Parent-child relations.
- Visibility.
- Occlusion.
- Skeleton topology.
- Coordinate system.
- Schema version.

Không tự áp dụng mapping Pose17 lên Face50 hoặc ngược lại.

Mỗi adapter phải expose interface thống nhất:

- `supports(task_schema)`
- `select_model(context)`
- `preprocess(images)`
- `predict(batch)`
- `convert_to_cvat(predictions)`
- `validate(proposals)`
- `refine(proposals)`
- `render_evidence(proposals)`

Tách domain logic khỏi CVAT API và UI.

## 6. ADAPTIVE IMAGE PREPROCESSING

Áp dụng với ảnh khó, không xử lý nặng tất cả ảnh.

Các phép biến đổi cần nghiên cứu:

- Brightness.
- Contrast.
- Saturation.
- Gamma.
- CLAHE.
- White balance khi phù hợp.
- Denoising khi cần.
- Rotation/orientation handling.

Đặc biệt chú ý ảnh tối, ảnh IR, kính mắt, râu, môi khép, profile và vùng lông mày.

Không chọn thông số bằng cảm tính.

Thiết kế preprocessing candidate search giới hạn ngân sách.

Ví dụ các bước:

1. Đo image statistics.
2. Phát hiện low-light/low-contrast.
3. Chạy baseline inference trên ảnh gốc.
4. Chọn tối đa một số phép biến đổi có cơ sở.
5. So sánh proposal bằng quality evaluator.
6. Chỉ áp dụng kết quả tốt hơn nếu có bằng chứng.
7. Lưu preprocessing recipe và kết quả.

Bắt buộc phân biệt ảnh cải thiện chỉ để HIỂN THỊ với ảnh cải thiện dùng làm INPUT INFERENCE.

Không chỉnh pixel ảnh nguồn trên CVAT.

Khi hiển thị evidence, cho phép người dùng chuyển giữa Original / Enhanced / Overlay.

Đối với Face50, tái sử dụng:

- CLAHE + Gamma.
- FAN2D.
- BrowEdgeSnap.
- InnerLipClamp.
- Region-based refinement.
- Geometry validation.

Không mặc định coi cấu hình đã thành công trên G06_091.jpg là tốt cho mọi ảnh.

Sử dụng nhiều ảnh kiểm chứng, đặc biệt ca profile, IR và lông mày khó.

## 7. AUTO MODEL SELECTION

CLI phải tự nhận diện annotation type và chọn pipeline phù hợp.

Selection dựa trên:

- CVAT label schema.
- Geometry type.
- Skeleton definition.
- Dataset characteristics.
- Hardware capacity.
- Existing model availability.
- Measured quality.
- Inference speed.
- Memory footprint.

Không chỉ dựa vào tên task.

Nếu không xác định được schema, dừng ở inspect/review thay vì đoán sai rồi ghi annotation.

Tạo model selection policy có thể thay đổi bằng config.

Cho phép người dùng override model.

## 8. DRAFT-FIRST, PRESERVE HUMAN EDITS

Mặc định:

**Generate proposals first. Never overwrite human edits automatically.**

Pipeline:

CVAT Discovery
→ Snapshot
→ Image Cache
→ Model Selection
→ Adaptive Preprocessing
→ Batch Inference
→ Proposal Generation
→ Geometry QC
→ Visual QC
→ Selective Refinement
→ Draft Storage
→ Evidence
→ Optional Safe Sync

Đối với annotation đang có:

- So sánh proposal với live snapshot.
- Ưu tiên giữ điểm/shape đã được người dùng sửa.
- Không thay đổi vùng PASS chỉ vì model đề xuất khác.
- Không tự đặt lại toàn bộ skeleton.
- Chỉ đề xuất sửa region có bằng chứng lỗi.
- Chỉ PATCH thay đổi tối thiểu.

Lưu riêng:

- Original annotation.
- Model proposal.
- Refined proposal.
- User-reviewed revision.
- CVAT synchronized revision.

Mọi proposal phải có provenance:

- Model ID/version.
- Image hash.
- Preprocessing parameters.
- Confidence/quality signals.
- Timestamp.
- Rule decisions.
- Run ID.

Nếu định dạng CVAT yêu cầu `source="manual"` cho annotation được tạo từ pipeline local, chỉ sử dụng khi tương thích và hợp lệ. Luôn giữ provenance thực tế là AI-generated draft ở metadata/history của công cụ. Không được tuyên bố người đã duyệt khi chưa có xác nhận.

## 9. QUALITY GATE & HUMAN REVIEW

Quality Gate phải phân biệt:

- Geometry valid.
- Image alignment good.
- Schema valid.
- Confidence sufficient.
- No regression.
- Human approved.

Geometry PASS không đồng nghĩa annotation đúng theo ảnh.

Tạo 3 mức:

**PASS**: đạt đầy đủ tiêu chí của policy.

**REFINE**: có khả năng tự sửa cục bộ.

**REVIEW**: thiếu bằng chứng, nhiều model bất đồng, lỗi khó hoặc không đạt quality threshold.

Giới hạn tối đa 2–3 refinement attempts cho mỗi region.

Không chạy vô hạn.

Dành riêng cho Face50:

- Brow edge alignment.
- Eye boundary.
- Nose alignment.
- Outer lip.
- Inner lip topology.
- Occluded landmark handling.
- Profile-aware geometry.
- Left/right consistency.
- Landmark movement regression.

Dành riêng cho Segmentation:

- Boundary bleed.
- Speckle.
- Self-intersecting polygon.
- Uncovered regions.
- Invalid overlaps.

Dành riêng cho BBox:

- Missing objects.
- Wrong class.
- Oversized box.
- Duplicate box.
- Occlusion errors.

Dành riêng cho Tracking:

- ID switches.
- Broken tracks.
- Temporal jumps.

## 10. AUTOMATED REVIEW IN VS CODE

Người dùng yêu cầu: nếu cần review ảnh, phải tự động mở evidence trong VS Code.

Triển khai:

- Render BEFORE / AFTER / OVERLAY.
- Render crop zoom cho vùng lỗi.
- Có ảnh Original và Enhanced.
- Hiển thị frame ID, label, affected points.
- Chỉ hiện tối đa 5 ca ưu tiên cao nhất mỗi lượt review.
- Tạo HTML review dashboard hoặc contact sheet.
- Gọi VS Code bằng cách được kiểm chứng trên Windows, ưu tiên `code.cmd --reuse-window`.

Không chỉ trả đường dẫn ảnh rồi yêu cầu người dùng tự mở.

Nếu mở thất bại, báo rõ command và lỗi.

Không hiển thị đánh giá mơ hồ như "đã đẹp" mà không có evidence.

Review dashboard phải có khả năng ghi nhận quyết định của người dùng cho từng proposal.

## 11. CLAUDE / ANTIGRAVITY ORCHESTRATION

Thiết kế orchestration theo nguyên tắc:

**Python does the work. AI agents handle exceptions.**

Không dùng Claude gọi inference từng ảnh.

Tạo optional `AgentProvider` interface để hỗ trợ:

- None / deterministic only.
- Claude CLI adapter khi đã cài đặt và xác thực.
- Antigravity-compatible workflow/skill.
- Provider khác về sau.

Không giả định Antigravity hoặc Claude CLI có API lập trình công khai. Kiểm tra command, khả năng gọi và điều kiện sử dụng thực tế trước khi viết adapter.

Nếu không thể gọi agent tự động từ CLI một cách được hỗ trợ, chuyển sang deterministic mode và tạo review packet để IDE agent tiếp tục; không giả lập integration.

Agent chỉ nên được gọi khi:

- Chưa hiểu task schema.
- Model inference thất bại.
- Có lỗi khó chưa thể sửa bằng code.
- Cần visual reasoning độc lập.
- Cần nghiên cứu model mới.
- Cần sửa bug trong pipeline.

Ưu tiên 0 agent calls cho happy path.

Đặt budget:

- Maximum agent calls per run.
- Maximum iterations.
- Maximum tokens.
- Maximum runtime.
- Maximum concurrent workers.

Không spawn 10 agent mặc định.

Không cho agent viết trực tiếp CVAT qua browser.

Chỉ một Sync Controller có quyền ghi.

## 12. SAFE SYNC & RECOVERY

Dry-run là mặc định.

Trước mỗi lần ghi:

1. Backup trạng thái live.
2. Kiểm tra revision hoặc snapshot fingerprint.
3. Kiểm tra shape ID.
4. Kiểm tra label ID.
5. Kiểm tra permission.
6. Kiểm tra quality gate.
7. Kiểm tra human-edit preservation.
8. Kiểm tra diff chính xác.

Chỉ sync nếu user đã chủ động chọn hoặc cấu hình chính sách auto-sync cho phạm vi đó.

Sau khi ghi:

- Read-back.
- Compare payload.
- Verify counts and geometry.
- Record mutation log.
- Rollback nếu an toàn và cần thiết.

Không rollback mù nếu có concurrent edit.

Hỗ trợ retry có giới hạn cho network failure.

Mọi run phải idempotent và resumable.

## 13. CLEAN ARCHITECTURE

Đề xuất tổ chức package như sau, nhưng ưu tiên refactor code hiện có:

```text
src/cvat_tools/
  factory/
    cli/
    core/
    cvat/
    auth/
    discovery/
    registry/
    adapters/
      detection/
      segmentation/
      tracking/
      skeleton/
    preprocessing/
    inference/
    quality/
    refinement/
    orchestration/
    review/
    sync/
    storage/
    reporting/
```

Không tạo module rỗng để đủ cây thư mục.

Không tạo framework phức tạp khi chưa có use case.

Áp dụng:

- Type hints.
- Pydantic/dataclasses nếu phù hợp.
- Dependency injection tối thiểu.
- Explicit errors.
- Structured logging.
- SQLite state store.
- Atomic file writes.
- Schema/versioned config.
- Unit/integration tests.
- CLI smoke tests.

Giữ backward compatibility cho command V2 nếu khả thi.

Không copy-paste engine Pose17/Face50 vào adapter mới.

Không tạo hai source of truth cho cùng một logic.

## 14. TEST MATRIX

Tạo test cho:

1. PAT configuration.
2. Expired/invalid PAT.
3. Role detection.
4. Read-only user.
5. Job discovery.
6. Taxonomy mismatch.
7. Model registry.
8. Cached model reuse.
9. Missing model.
10. CPU fallback.
11. Mixed image quality.
12. Pose17 schema.
13. Face50 schema.
14. BBox conversion.
15. Segmentation conversion.
16. Tracking consistency.
17. Human edit preservation.
18. Concurrent CVAT edit.
19. Failed/partial sync.
20. Read-back mismatch.
21. Rollback conflict.
22. Resume after crash.
23. Repeated run idempotence.
24. VS Code evidence opening.
25. CLI installation and entrypoint.

Dùng fixtures/mocks cho destructive tests.

Không dùng live CVAT như test database.

Chỉ thực hiện read-only smoke test trên Job 1623 và 1975.

## 15. BENCHMARK

Chạy benchmark trên một tập dữ liệu đại diện, gồm ảnh dễ và ảnh khó.

Đo:

- Total runtime.
- Model loading time.
- Preprocessing latency.
- Inference latency.
- QC latency.
- Frames/minute.
- Peak RAM/VRAM.
- Agent calls.
- Token input/output nếu có.
- Cache hit rate.
- Proposal acceptance.
- Review rate.
- Geometry failures.
- Visual alignment failures.
- Regression count.

So sánh V2 với V3 trên cùng workload.

Không lấy tốc độ chạy cached/no-op để tuyên bố tăng tốc inference.

Không tuyên bố đạt 90% chất lượng nếu chưa có human-labeled ground truth hoặc independent assessment phù hợp.

## 16. EXECUTION ORDER

### Phase 0 — Audit

Kiểm kê repository, model, scripts, skill, cache, GPU/CPU và package dependencies.

Tạo `MODEL_INVENTORY.md` và `REUSE_MAP.md`.

Phân loại:

- REUSE.
- REFACTOR.
- REPLACE.
- DEPRECATE.

### Phase 1 — CLI Foundation

Tạo package CLI thực sự cài đặt được.

Hoàn thiện doctor, setup, discover, inspect, models list.

Test ngay bằng môi trường hiện tại.

### Phase 2 — Universal Engine

Di chuyển logic dùng chung.

Tích hợp Face50 và Pose17.

Tạo interface cho BBox, Segmentation, Tracking.

Hoàn thiện ít nhất một vertical slice chạy thật cho mỗi type có sẵn model/test fixture. Nếu chưa có model thì hỗ trợ capability discovery, validation và báo rõ chưa có inference adapter.

### Phase 3 — Model Optimization

Triển khai registry, model selection, preprocessing strategy và batch inference.

Tái sử dụng model weights đã tải.

### Phase 4 — Review & Quality

Triển khai QC, evidence, review queue và VS Code integration.

### Phase 5 — Safe Synchronization

Tích hợp backup, conflict detection, PATCH, read-back, rollback.

Mặc định dry-run.

### Phase 6 — Verification

Chạy toàn bộ test, benchmark và read-only CVAT smoke test.

Sửa lỗi phát hiện được.

### Phase 7 — Packaging

Cập nhật một skill tái sử dụng cho Antigravity.

Cập nhật README và command examples.

Tạo quickstart Windows PowerShell.

Không tạo nhiều skill trùng nhau.

## 17. DEFINITION OF DONE

Chỉ đánh dấu DONE khi:

- `cvat-factory --help` chạy được.
- `cvat-factory doctor` hoạt động.
- PAT được cấu hình an toàn.
- Discover/inspect đọc được CVAT.
- Model registry tìm được các model thực sự tồn tại.
- Face50/Pose17 chạy qua unified CLI.
- Ít nhất một lệnh dry-run tạo được annotation proposal thực từ model có sẵn.
- Evidence được tạo và tự mở trong VS Code.
- Tests pass.
- Không ghi đè human edits.
- CLI có thể resume.
- Có benchmark đo thật.
- Có command chạy lại và tài liệu sử dụng.

Chỉ báo hỗ trợ BBox/Segmentation/Tracking hoàn chỉnh khi đã chạy integration test tương ứng.

Không bắt người dùng cung cấp lại thông tin đã nằm trong config hoặc repository.

Không hỏi xác nhận cho refactor, test, dry-run và thao tác local không phá dữ liệu.

Các hoạt động ghi annotation live phải tuân thủ policy sync đã cấu hình; mặc định không ghi.

## 18. FINAL OUTPUT

Trả lời ngắn bằng tiếng Việt:

1. CLI đã cài đặt được chưa?
2. Những model nào đã tìm thấy, path và khả năng sử dụng?
3. Từng annotation type hỗ trợ đến mức nào?
4. Những test nào đã pass/fail?
5. Benchmark V2 vs V3 thực tế.
6. Những tính năng còn chưa hoàn thiện.
7. Command duy nhất người dùng cần chạy.
8. Evidence đã mở trong VS Code ở đâu?

**BẮT ĐẦU THỰC THI NGAY. KHÔNG CHỈ TẠO KẾ HOẠCH. TẬP TRUNG CODE CHẠY ĐƯỢC, TÁI SỬ DỤNG MODEL CŨ, GIỮ NGUYÊN HUMAN EDITS VÀ GIẢM TỐI ĐA THAO TÁC NGƯỜI DÙNG.**