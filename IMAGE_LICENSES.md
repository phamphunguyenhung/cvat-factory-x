# Image, Visual Asset & Dataset Licenses — CVAT Factory X

Tài liệu này ghi nhận rõ ràng nguồn gốc (source), bản quyền (copyright) và giấy phép sử dụng (license) cho toàn bộ tài nguyên hình ảnh, sơ đồ kỹ thuật, slide trình bày và bộ dữ liệu mẫu (sample dataset) có trong repository **CVAT Factory X**.

> Mã nguồn phần mềm của dự án được phát hành theo giấy phép [**MIT License**](LICENSE). Các tài nguyên dữ liệu bên thứ ba giữ nguyên điều khoản bản quyền của tác giả gốc.

---

## 1. Bảng tổng hợp bản quyền tài nguyên (Asset & Dataset License Summary)

| Tài nguyên / Thư mục | Loại tài nguyên | Nguồn gốc (Source / Provenance) | Giấy phép (License) | Ghi chú |
| :--- | :--- | :--- | :--- | :--- |
| `docs/assets/hero-banner.png` | Technical Illustration & Banner | Project-owned technical illustration generated for CVAT Factory X + custom typography | **MIT License** | Thuộc bản quyền dự án CVAT Factory X ([`LICENSE`](LICENSE)) |
| `docs/assets/pipeline-7-stages.png` | Architecture / Pipeline Diagram | Generated deterministically from [`scripts/generate_readme_diagrams.py`](scripts/generate_readme_diagrams.py) | **MIT License** | Tự động sinh (deterministic rendering) từ mã nguồn Python của dự án |
| `docs/assets/scale-workload-split.png` | Workload Scaling Diagram | Generated deterministically from [`scripts/generate_readme_diagrams.py`](scripts/generate_readme_diagrams.py) | **MIT License** | Minh họa tách biệt Machine Workload vs. Human Workload |
| `docs/assets/architecture-planes.png` | System Architecture Diagram | Generated deterministically from [`scripts/generate_readme_diagrams.py`](scripts/generate_readme_diagrams.py) | **MIT License** | Minh họa Control Plane vs. Data Plane trong `release_control` |
| `docs/assets/day10-poc-loop.png` | Day 10 PoC Workflow Diagram | Generated deterministically from [`scripts/generate_readme_diagrams.py`](scripts/generate_readme_diagrams.py) | **MIT License** | Minh họa chu trình PoC: Load → Inject → HOLD → Trace → Restore |
| `docs/assets/roadmap-phases.png` | Product Roadmap Diagram | Generated deterministically from [`scripts/generate_readme_diagrams.py`](scripts/generate_readme_diagrams.py) | **MIT License** | Minh họa lộ trình 5 giai đoạn phát triển |
| `docs/slides/slide_01.jpg` – `slide_10.jpg` | Presentation Slide Images (10 slides) | Original project presentation slides — CVAT Factory X | **CC BY 4.0** (Project Presentation Deck) | Slide trình bày gốc của nhóm phát triển CVAT Factory X |
| [`docs/CVAT_Factory_X_Presentation.pdf`](docs/CVAT_Factory_X_Presentation.pdf) | Presentation Deck (PDF) | Original project presentation deck — CVAT Factory X | **CC BY 4.0** (Project Presentation Deck) | Bản PDF tổng hợp bộ slide thuyết trình của dự án |
| `demo/coco128_clean/` (images, labels, masks, metadata) | Sample Dataset (Images & Annotations) | Derived from **MS COCO val2017** ([`cocodataset.org`](https://cocodataset.org)) | **CC BY 4.0** (annotations) & **Flickr Terms of Use** (images) | Xem chi tiết mục 3 bên dưới và [`demo/LICENSE_NOTE.md`](demo/LICENSE_NOTE.md) |

---

## 2. Chi tiết tài nguyên sở hữu bởi dự án (Project-Owned Visual Assets)

### 2.1. Hero Banner & Sơ đồ kỹ thuật (`docs/assets/`)

- **`docs/assets/hero-banner.png`**: Ảnh minh họa kỹ thuật (technical illustration) được tạo riêng cho CVAT Factory X kết hợp typography tùy chỉnh. Phát hành dưới giấy phép **MIT License** cùng repository.
- **Sơ đồ kỹ thuật (`docs/assets/*.png`)**:
  - `docs/assets/pipeline-7-stages.png`
  - `docs/assets/scale-workload-split.png`
  - `docs/assets/architecture-planes.png`
  - `docs/assets/day10-poc-loop.png`
  - `docs/assets/roadmap-phases.png`
  
  Các tệp PNG trên được sinh ra tất định (generated deterministically) từ script [`scripts/generate_readme_diagrams.py`](scripts/generate_readme_diagrams.py) thuộc dự án CVAT Factory X và tuân theo **MIT License**.

### 2.2. Bộ slide & Tài liệu thuyết trình (`docs/slides/` & PDF Deck)

- **`docs/slides/slide_01.jpg` – `slide_10.jpg`** và **[`docs/CVAT_Factory_X_Presentation.pdf`](docs/CVAT_Factory_X_Presentation.pdf)**:
  - Là bộ slide thuyết trình gốc (Original Project Presentation Deck) do đội ngũ **CVAT Factory X / VinUni AI20K L2/L3 Contributors** biên soạn.
  - Phát hành theo giấy phép **Creative Commons Attribution 4.0 International (CC BY 4.0)** — cho phép chia sẻ và chỉnh sửa kèm ghi nhận nguồn tác giả dự án.

---

## 3. Chi tiết bộ dữ liệu mẫu bên thứ ba (`demo/coco128_clean/`)

Thư mục `demo/coco128_clean/` chứa dữ liệu mẫu phục vụ minh họa và kiểm thử PoC offline (images, YOLO labels, segmentation masks, `manifest.json`, `panoptic.json`, v.v.):

- **Nguồn gốc (Source):** Trích xuất và chuyển đổi từ tập dữ liệu công khai **MS COCO val2017** (`https://cocodataset.org`).
- **Giấy phép Annotations:** Bộ nhãn (annotations) thuộc MS COCO được cấp phép theo **CC BY 4.0** ([Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)).
- **Giấy phép Hình ảnh (Images):** Các tệp ảnh giữ nguyên điều khoản bản quyền gốc trên **Flickr** theo quy định tại **COCO Terms of Use** (`https://cocodataset.org/#termsofuse`).
- **Trích dẫn bắt buộc (Attribution):**
  > Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. *"Microsoft COCO: Common Objects in Context."* European Conference on Computer Vision (ECCV), 2014.
- **Ghi chú dữ liệu tổng hợp (Synthetic Lab Metadata):** Các tệp metadata điều phối trong lab (`jobs.csv`, `manifest.json`, `manifest.lab.json`, `metadata.json`, `release.json`, scene/job/annotator metadata) là dữ liệu giả lập phục vụ thực hành theo ghi chú tại [`demo/LICENSE_NOTE.md`](demo/LICENSE_NOTE.md).

---

## 4. Liên kết liên quan

- Giấy phép mã nguồn: [`LICENSE`](LICENSE)
- Ghi chú bản quyền dataset demo: [`demo/LICENSE_NOTE.md`](demo/LICENSE_NOTE.md)
- Tài liệu tổng quan (English): [`README.md`](README.md)
- Tài liệu tổng quan (Tiếng Việt): [`README.vi.md`](README.vi.md)
