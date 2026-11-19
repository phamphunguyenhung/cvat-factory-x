# Submission — bài nhóm

## Repo bài nộp

Mỗi nhóm **một repo public** (tạo từ repo đề), đặt tên theo mẫu:

| Mẫu tên repo | Ví dụ |
|---|---|
| `K5-L23-DAY10-NHOM[XX]-Release` | `K5-L23-DAY10-NHOM03-Release` |

Theo hướng dẫn lớp nếu giảng viên đặt quy ước khác.

## Trước khi nộp

1. Đổi dòng tiêu đề `# Day 10 Lab — ...` trong `README.md` (ngay dưới ảnh hero) thành `# Day 10 Lab — Nhóm <XX>`.
2. Chép bài làm của seed luyện tập: `python scripts/export_submission.py --run runs/r<seed>`
3. `python scripts/check_handover.py --run runs/r<seed>` không còn FAIL.
4. Bài đề ẩn nằm trong `submission/de_an/` (không commit thư mục `delivery/` của đề).
5. Nộp link repo lên LMS **https://aithucchien.vinuni.edu.vn** hoặc kênh giảng viên/coach chỉ định.

## Cấu trúc bắt buộc

```text
README.md                           # tiêu đề: # Day 10 Lab — Nhóm <XX>
validator/
└── validate.py                     # NHÓM TỰ VIẾT: validate.py <delivery> --out errors.json [--candidates c.json]
submission/
├── TEAM.md                         # thành viên, MSHV, ai làm phần nào, seed luyện tập
├── REPORT.md                       # số liệu, câu hỏi trong GUIDE, bug tìm thấy (nếu có)
├── prompts.md                      # prompt AI, chỗ AI sai, cách kiểm lại
├── lineage.jsonl                   # chép từ runs/lineage.jsonl
├── r<seed>/                        # SEED LUYỆN TẬP
│   ├── errors.json
│   ├── candidates.json
│   ├── review_log.csv
│   ├── trace.json
│   ├── security.md
│   └── handover/
│       ├── <release_id>_v2/        # images/, labels/, manifest.csv, label_summary.csv,
│       │                           # truyvet.csv, README.txt, rework_queue.csv
│       └── <release_id>_v2.sha256  # KHÔNG commit .zip
└── de_an/                          # ĐỀ ẨN
    ├── errors.json
    ├── candidates.json
    ├── review_log.csv
    └── trace.json
```

Ảnh trong `handover/<release_id>_v2/images/` là ảnh COCO (khoảng 40 MB). Nếu lớp giới hạn dung lượng repo, giảng viên sẽ báo cách khác.

## Checklist trước khi nộp

- [ ] Validator chỉ đọc `delivery/`, chạy được trên lô sạch mà không báo lỗi luật nào.
- [ ] Mọi lỗi gán nhãn trong `errors.json` có dòng `confirmed` trong `review_log.csv`, ghi rõ thấy gì.
- [ ] v1 giữ nguyên; v2 có manifest + sha256, label summary, truy vết, rework queue, README.
- [ ] Lineage có HOLD v1 và release v2 với parent.
- [ ] `check_handover.py` không còn FAIL.
- [ ] `trace.json` có `tier_a`, `tier_b`, `notify`.
- [ ] `security.md` đủ ba mục.
- [ ] Mỗi người có ít nhất 2 commit có nội dung.
