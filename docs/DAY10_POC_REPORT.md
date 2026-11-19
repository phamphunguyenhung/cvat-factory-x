# Báo cáo Tổng kết Proof of Concept — CVAT Factory X (Day 10)

**Dự án:** CVAT Factory X — Autonomous Driving Data Factory & Release Control Tower  
**Bối cảnh:** VinUni AI20K L2/L3 — Day 10: Data Handoff & Traceability  
**Repository Public:** [https://github.com/phamphunguyenhung/cvat-factory-x](https://github.com/phamphunguyenhung/cvat-factory-x)  
**Ngày lập báo cáo:** 2026-11-19  

---

## 1. Tóm tắt điều hành (Executive Summary)

PoC Day 10 đã chứng minh thành công năng lực nền tảng bắt buộc trước khi mở rộng quy mô: **dữ liệu đầu ra có thể được kiểm tra tự động theo hợp đồng, chặn phát hành khi có lỗi (`HOLD`), truy vết bán kính ảnh hưởng qua Lineage và khôi phục nguyên trạng bằng SHA-256 Time Machine Restore.**

- **Bộ kiểm thử tự động (`python -m pytest tests/ -q`):** **26/26 passed** (`0 failed`).
- **Luồng Demo End-to-End (`python -m release_control demo`):** **ĐẠT (`KET QUA: DAT`)** — chạy trọn vẹn chu trình `Load -> Inject (T6+T5) -> HOLD -> Trace Lineage -> Restore SHA-256 -> VALIDATED`.
- **Bộ tài liệu & Trình bày:** Hoàn thiện `README.md`, `README.vi.md`, `docs/PIPELINE_AND_MODULES.md`, `docs/DAY10_POC_GUIDE.md`, `docs/GUIDELINE_SSOT_WORKFLOW.md`, `IMAGE_LICENSES.md`, 6 sơ đồ kiến trúc (`docs/assets/`) và slide deck PDF 10 trang (`docs/CVAT_Factory_X_Presentation.pdf`).

---

## 2. Kết quả kiểm chứng PoC & Benchmark (`reports/`)

Số liệu ghi nhận trực tiếp từ `reports/FAULT_BENCHMARK.json` và `reports/RELEASE_EVIDENCE.json`:

| Tiêu chí kiểm chứng | Kết quả thực nghiệm | Ghi chú |
| :--- | :---: | :--- |
| **Test suite (`tests/`)** | **26 passed** (63.64s) | Toàn bộ unit & integration tests đạt |
| **Fault Recall (`T1`–`T8`)** | **100%** (40/40 fault runs) | Dev seeds `[42, 43]` & held-out seeds `[101, 102, 103]` |
| **Critical Fault Recall** | **1.0** (100%) | Không bỏ sót lỗi nghiêm trọng |
| **False Hold Rate** | **0.0%** | Không chặn nhầm dữ liệu sạch |
| **False Release Rate** | **0.0%** | Không để lọt dữ liệu lỗi qua Quality Gate |
| **Metamorphic Tests** | **All Passed** | Bất biến thứ tự manifest; 0 cảnh báo giả trên bản sạch |
| **Độ trễ kiểm định trung bình** | **478.54 ms** | Phù hợp tích hợp kiểm tra tự động |
| **Release Passport Evidence** | **VALIDATED** | `reports/RELEASE_EVIDENCE.json` (SHA-256 `1b664ab18ebf...`) |

---

## 3. Đánh giá thẳng thắn: Đã đạt vs Chưa đạt

### Đã đạt trong Day 10 PoC
- Kiểm soát hợp đồng dữ liệu (`schema.yaml`, `taxonomy.yaml`, `release_policy.yaml`) và state machine `DRAFT → VALIDATED → APPROVED → PUBLISHED → ACCEPTED`.
- Bảo vệ dữ liệu nguồn (`coco250`, `coco128_clean`, CVAT job caches) ở chế độ **chỉ đọc**; mọi thao tác tiêm lỗi và khôi phục chạy cô lập trong `artifacts/`.
- Đồ thị Lineage có bằng chứng (`openlineage.jsonl` + SQLite + MLflow), tách bạch rõ cạnh xác thực và cạnh mô phỏng (`simulated_demo`).

### Chưa đạt / Phạm vi tiếp theo (Phase 2–5)
- **Mục tiêu 2 triệu frames / 14 ngày:** Chưa benchmark trên hạ tầng đa worker thực tế (demo chạy trên tập mẫu nhỏ; `release_control/scale/` mới ở mức khung).
- **Luồng Guideline SSOT liên tục cho annotator tay:** Mới có `guideline_ref` tĩnh và kiểm định hậu kiểm `T8_GUIDELINE_MISMATCH`; chưa có cơ chế tự động push diff guideline lên CVAT và bắt buộc Annotator ACK (đã đặc tả trong `docs/GUIDELINE_SSOT_WORKFLOW.md`).

---

## 4. Thống kê Token & Chi phí Vận hành AI (Phiên Day 10)

Số liệu được trích xuất thụ động từ log thực thi của **Claude Code** và **Antigravity**:

### 4.1. Claude Code (Triển khai Core Features & Tests)
Nguồn: `C:\Users\ppnh1\.claude\projects\D--06-Learning-AIA-VINGROUP-learning-day10-projects\` (Model: `claude-sonnet-5-5`, 1 session chính + 14 subagents):

| Hạng mục | Main Session | 14 Subagents | Tổng cộng |
| :--- | ---: | ---: | ---: |
| **API Calls** | 104 | 312 | **416** |
| **Input Tokens (uncached)** | 280 | 626 | **906** |
| **Cache Creation Tokens** | 331,010 | 1,853,416 | **2,184,426** |
| **Cache Read Tokens** | 12,466,717 | 19,146,330 | **31,613,047** |
| **Output Tokens** | 121,481 | 476,024 | **597,505** |
| **Tổng tất cả Tokens** | **12,919,488** | **21,476,396** | **34,395,884 (~34.40M)** |

- **Tổng token mới (không tính Cache Read):** `906 + 2,184,426 + 597,505 =` **2,782,837 tokens (~2.78M)**.

### 4.2. Antigravity (Tài liệu, Sơ đồ, PDF, Licensing, GitHub SEO & Verification)
Nguồn: `transcript_full.jsonl` của 1 session chính (Gemini 4 Hydrogen Medium) + 5 subagents.  
*(Lưu ý: `transcript_full.jsonl` của Antigravity không ghi trường billing `usage` token từ API; bảng dưới ghi nhận số đo ký tự thực tế trong log).*

| Thành phần | Steps | Thinking (ký tự) | Content + Tool Log (ký tự) | Tổng ký tự |
| :--- | ---: | ---: | ---: | ---: |
| **Main Session** | 93 | 14,103 | 91,907 | **106,010** |
| **5 Subagents** | 96 | 9,120 | 109,765 | **118,885** |
| **Tổng cộng** | **189** | **23,223** | **201,672** | **224,895** |

- **Quy đổi ước lượng nội dung log (~4 ký tự ≈ 1 token):** **~56,224 tokens** *(chưa bao gồm system prompt, tool schema và context lặp lại qua các turn)*.
