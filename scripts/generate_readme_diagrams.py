"""Generate crisp Light-Mode Vietnamese technical diagrams for CVAT Factory X READMEs."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BG = "#F8FAFC"
CARD_BG = "#FFFFFF"
BORDER = "#CBD5E1"
TEXT_DARK = "#0F172A"
TEXT_MUTED = "#475569"
BLUE = "#2563EB"
BLUE_SOFT = "#EFF6FF"
INDIGO = "#4F46E5"
INDIGO_SOFT = "#EEF2FF"
EMERALD = "#059669"
EMERALD_SOFT = "#ECFDF5"
AMBER = "#D97706"
AMBER_SOFT = "#FFFBEB"
RED = "#DC2626"
RED_SOFT = "#FEF2F2"
SLATE_SOFT = "#F1F5F9"


def _box(ax, x, y, w, h, fc=CARD_BG, ec=BORDER, lw=1.5, r=0.015):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        transform=ax.transAxes,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def _arrow(ax, x1, y1, x2, y2, color="#64748B", lw=1.8, style="-|>", rad=0.0):
    arr = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        transform=ax.transAxes,
        zorder=3,
    )
    ax.add_patch(arr)


def _new_canvas(w_in=14, h_in=6.2):
    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=220)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def draw_pipeline_7_stages():
    fig, ax = _new_canvas(15, 6.4)
    ax.text(0.03, 0.93, "AUTONOMOUS DRIVING DATA PIPELINE — 7 GIAI ĐOẠN KHÉP KÍN",
            fontsize=16, fontweight="bold", color=TEXT_DARK, transform=ax.transAxes)
    ax.text(0.03, 0.88, "Ingest → Pre-label → Risk Routing → Human Collaboration → Quality Gate → Versioning & Lineage → Model Feedback",
            fontsize=10.5, color=TEXT_MUTED, transform=ax.transAxes)

    stages = [
        ("1", "Data Ingestion\n& Governance", "Nguồn, scene, sensor,\nchecksum, chống leakage", BLUE, BLUE_SOFT),
        ("2", "Intelligent\nPre-labeling", "Detection, tracking,\nlưu model version & conf.", INDIGO, INDIGO_SOFT),
        ("3", "Risk & Uncertainty\nRouting", "Chấm điểm bất định,\nche khuất, thời tiết xấu", AMBER, AMBER_SOFT),
        ("4", "Human\nCollaboration", "CVAT review, phân công,\nguideline & audit trail", BLUE, BLUE_SOFT),
        ("5", "Quality Gate\n6 Lớp Kiểm Định", "Structural, Geometric,\nSemantic, Stat, QC, Privacy", RED, RED_SOFT),
        ("6", "Versioning,\nLineage & Release", "Snapshot SHA-256,\nContract, Approval, ACK", EMERALD, EMERALD_SOFT),
        ("7", "Model\nFeedback Loop", "Lỗi đánh giá model →\nhàng đợi chọn mẫu mới", INDIGO, INDIGO_SOFT),
    ]

    # Top row: stages 1..4, Bottom row: stages 7..5 (clockwise loop) or single clean row + feedback arc
    # Single clean row of 7 cards works great on 15in wide canvas!
    n = len(stages)
    margin = 0.028
    gap = 0.016
    card_w = (1.0 - 2 * margin - (n - 1) * gap) / n
    card_h = 0.44
    y0 = 0.34

    centers_x = []
    for i, (num, title, desc, col, soft) in enumerate(stages):
        x0 = margin + i * (card_w + gap)
        centers_x.append(x0 + card_w / 2)
        _box(ax, x0, y0, card_w, card_h, fc=CARD_BG, ec=col, lw=1.8, r=0.014)
        # Header pill
        _box(ax, x0 + 0.01, y0 + card_h - 0.085, card_w - 0.02, 0.068, fc=soft, ec="none", r=0.01)
        ax.text(x0 + card_w / 2, y0 + card_h - 0.051, f"STAGE {num}",
                ha="center", va="center", fontsize=8.8, fontweight="bold", color=col, transform=ax.transAxes)
        ax.text(x0 + card_w / 2, y0 + card_h - 0.165, title,
                ha="center", va="center", fontsize=10, fontweight="bold", color=TEXT_DARK, transform=ax.transAxes)
        ax.text(x0 + card_w / 2, y0 + 0.10, desc,
                ha="center", va="center", fontsize=8.3, color=TEXT_MUTED, linespacing=1.45, transform=ax.transAxes)
        if i < n - 1:
            _arrow(ax, x0 + card_w + 0.002, y0 + card_h / 2, x0 + card_w + gap - 0.002, y0 + card_h / 2, color="#475569", lw=1.8)

    # Bottom feedback bar from Stage 7 back to Stage 2/3
    _box(ax, 0.028, 0.08, 0.944, 0.14, fc=SLATE_SOFT, ec=BORDER, lw=1.2, r=0.012)
    ax.text(0.50, 0.165, "DATA-CENTRIC FEEDBACK LOOP: Kết quả đánh giá Model (miss pedestrian, false positive ban đêm) → Tự động ưu tiên mẫu cho vòng Annotation & QC tiếp theo",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color=INDIGO, transform=ax.transAxes)
    ax.text(0.50, 0.115, "Nguyên tắc cốt lõi:  AI-First (xử lý lặp lại quy mô lớn)   •   Human-on-the-Loop (tập trung ca rủi ro cao)   •   Traceable by Design (truy vết đầy đủ)",
            ha="center", va="center", fontsize=9.0, color=TEXT_MUTED, transform=ax.transAxes)

    _arrow(ax, centers_x[6], y0, centers_x[6], 0.225, color=INDIGO, lw=1.8)
    _arrow(ax, centers_x[2], 0.225, centers_x[2], y0, color=INDIGO, lw=1.8)

    fig.savefig(OUT_DIR / "pipeline-7-stages.png", bbox_inches="tight", pad_inches=0.18, facecolor=BG)
    plt.close(fig)


def draw_scale_workload_split():
    fig, ax = _new_canvas(14, 6.4)
    ax.text(0.03, 0.93, "BÀI TOÁN MỞ RỘNG: 10 ANNOTATORS XỬ LÝ 2.000.000 FRAMES TRONG 14 NGÀY",
            fontsize=15.5, fontweight="bold", color=TEXT_DARK, transform=ax.transAxes)
    ax.text(0.03, 0.88, "Kịch bản kiểm tra kiến trúc (Architecture Stress Scenario): 142.857 frames/ngày → Tách Machine Workload & Human Workload",
            fontsize=10.2, color=TEXT_MUTED, transform=ax.transAxes)

    # Top KPI strip
    kpis = [
        ("2.000.000", "Tổng số Frame mục tiêu", BLUE, BLUE_SOFT),
        ("14 Ngày", "Khung thời gian xử lý", INDIGO, INDIGO_SOFT),
        ("142.857 / ngày", "Thông lượng trung bình", AMBER, AMBER_SOFT),
        ("14.286 / người / ngày", "Nếu gán tay 100% (Bất khả thi)", RED, RED_SOFT),
    ]
    for i, (val, lbl, col, soft) in enumerate(kpis):
        x0 = 0.03 + i * 0.24
        _box(ax, x0, 0.69, 0.22, 0.14, fc=soft, ec=col, lw=1.4, r=0.012)
        ax.text(x0 + 0.11, 0.785, val, ha="center", va="center", fontsize=12.5, fontweight="bold", color=col, transform=ax.transAxes)
        ax.text(x0 + 0.11, 0.725, lbl, ha="center", va="center", fontsize=8.8, color=TEXT_DARK, transform=ax.transAxes)

    # Left: Machine Workload
    _box(ax, 0.03, 0.09, 0.45, 0.54, fc=CARD_BG, ec=BLUE, lw=1.8, r=0.014)
    _box(ax, 0.045, 0.545, 0.42, 0.065, fc=BLUE_SOFT, ec="none", r=0.01)
    ax.text(0.255, 0.577, "MACHINE WORKLOAD — Tự động hóa quy mô lớn",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color=BLUE, transform=ax.transAxes)
    machine_items = [
        "• Đọc, chuẩn hóa dữ liệu & kiểm tra checksum theo shard/batch",
        "• Pre-labeling: Object detection, segmentation, tracking tự động",
        "• Lọc frame trùng lặp hoặc ít thông tin mới (deduplication)",
        "• Tính Risk Score & Uncertainty để phân loại độ ưu tiên review",
        "• Chạy Quality Gate, đóng gói snapshot SHA-256 & xuất báo cáo",
    ]
    for idx, txt in enumerate(machine_items):
        ax.text(0.055, 0.47 - idx * 0.075, txt, ha="left", va="center", fontsize=9.3, color=TEXT_DARK, transform=ax.transAxes)

    # Right: Human Workload
    _box(ax, 0.52, 0.09, 0.45, 0.54, fc=CARD_BG, ec=EMERALD, lw=1.8, r=0.014)
    _box(ax, 0.535, 0.545, 0.42, 0.065, fc=EMERALD_SOFT, ec="none", r=0.01)
    ax.text(0.745, 0.577, "HUMAN WORKLOAD — 10 Thành viên tập trung ca khó",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color=EMERALD, transform=ax.transAxes)
    human_items = [
        "• Xử lý các frame có độ bất định cao hoặc model không chắc chắn",
        "• Hiệu chỉnh nhãn có ảnh hưởng lớn (người đi bộ, xe nhỏ ở xa)",
        "• Kiểm tra tình huống an toàn quan trọng (mưa, ban đêm, che khuất)",
        "• Phê duyệt mẫu QC, privacy gate & các trường hợp ngoại lệ",
        "• Cập nhật annotation guideline khi phát hiện loại lỗi mới",
    ]
    for idx, txt in enumerate(human_items):
        ax.text(0.545, 0.47 - idx * 0.075, txt, ha="left", va="center", fontsize=9.3, color=TEXT_DARK, transform=ax.transAxes)

    _arrow(ax, 0.483, 0.36, 0.517, 0.36, color=AMBER, lw=2.4)
    ax.text(0.50, 0.41, "Risk\nQueue", ha="center", va="center", fontsize=8.3, fontweight="bold", color=AMBER, transform=ax.transAxes)

    fig.savefig(OUT_DIR / "scale-workload-split.png", bbox_inches="tight", pad_inches=0.18, facecolor=BG)
    plt.close(fig)


def draw_architecture_planes():
    fig, ax = _new_canvas(14.5, 6.8)
    ax.text(0.03, 0.935, "KIẾN TRÚC TÁCH LỚP: DATA PLANE vs CONTROL PLANE (release_control/)",
            fontsize=15.5, fontweight="bold", color=TEXT_DARK, transform=ax.transAxes)
    ax.text(0.03, 0.89, "Thiết kế sẵn sàng mở rộng: Idempotent • Checkpoint/Resume • Immutable Snapshots • Content-Addressed Integrity • Least-Privilege",
            fontsize=10.0, color=TEXT_MUTED, transform=ax.transAxes)

    # Data Plane (Left)
    _box(ax, 0.03, 0.08, 0.36, 0.76, fc=CARD_BG, ec=BLUE, lw=1.8, r=0.015)
    _box(ax, 0.045, 0.765, 0.33, 0.058, fc=BLUE_SOFT, ec="none", r=0.01)
    ax.text(0.21, 0.794, "DATA PLANE (Xử lý lô & Lưu trữ)", ha="center", va="center", fontsize=10.5, fontweight="bold", color=BLUE, transform=ax.transAxes)

    dp_blocks = [
        ("Nguồn dữ liệu Read-Only", "COCO250 / COCO128, CVAT Jobs (1975, 1623)\nKhông sửa đổi dữ liệu gốc", 0.61),
        ("Working Copies & Shards", "artifacts/working/ — Bản sao cô lập\nXử lý theo shard & batch, checkpoint/resume", 0.43),
        ("Immutable Snapshots", "artifacts/snapshots/ — SHA-256 Digest\nContent-addressed, không sửa trực tiếp", 0.25),
        ("Evidence & Reports", "reports/RELEASE_EVIDENCE.json\nreports/FAULT_BENCHMARK.json", 0.10),
    ]
    for title, desc, y0 in dp_blocks:
        _box(ax, 0.05, y0, 0.32, 0.125, fc=SLATE_SOFT, ec=BORDER, lw=1.2, r=0.01)
        ax.text(0.21, y0 + 0.088, title, ha="center", va="center", fontsize=9.8, fontweight="bold", color=TEXT_DARK, transform=ax.transAxes)
        ax.text(0.21, y0 + 0.038, desc, ha="center", va="center", fontsize=8.3, color=TEXT_MUTED, linespacing=1.35, transform=ax.transAxes)

    # Control Plane (Right)
    _box(ax, 0.43, 0.08, 0.54, 0.76, fc=CARD_BG, ec=EMERALD, lw=1.8, r=0.015)
    _box(ax, 0.445, 0.765, 0.51, 0.058, fc=EMERALD_SOFT, ec="none", r=0.01)
    ax.text(0.70, 0.794, "CONTROL PLANE (Điều phối, Kiểm định & Truy vết)", ha="center", va="center", fontsize=10.5, fontweight="bold", color=EMERALD, transform=ax.transAxes)

    cp_mods = [
        ("adapters/", "CvatConnector (read-only)\nDatumaro & MLflow Adapter", BLUE, BLUE_SOFT),
        ("contracts/", "Schema, Taxonomy,\nRelease Policy YAML", INDIGO, INDIGO_SOFT),
        ("validators/", "Structural, Geometry,\nSemantic, Privacy (T1–T8)", RED, RED_SOFT),
        ("fault_lab/", "FaultInjector (seed) &\nFaultBenchmarker", AMBER, AMBER_SOFT),
        ("versions/", "SnapshotManager &\nTime Machine Restore", EMERALD, EMERALD_SOFT),
        ("lineage/", "OpenLineage Events &\nBFS Impact / Blast Radius", INDIGO, INDIGO_SOFT),
        ("release/", "State Machine Gate &\nReleasePassport Evidence", EMERALD, EMERALD_SOFT),
        ("scale/", "Sharding, Risk Scoring,\nReview Queue & Capacity", BLUE, BLUE_SOFT),
        ("dashboard/ & cli.py", "FastAPI Control Tower\n& cvat-release CLI", TEXT_DARK, SLATE_SOFT),
    ]
    for idx, (mname, mdesc, col, soft) in enumerate(cp_mods):
        r = idx // 3
        c = idx % 3
        mx = 0.448 + c * 0.172
        my = 0.545 - r * 0.215
        _box(ax, mx, my, 0.16, 0.18, fc=soft, ec=col, lw=1.3, r=0.01)
        ax.text(mx + 0.08, my + 0.135, mname, ha="center", va="center", fontsize=9.3, fontweight="bold", color=col, transform=ax.transAxes)
        ax.text(mx + 0.08, my + 0.06, mdesc, ha="center", va="center", fontsize=8.0, color=TEXT_DARK, linespacing=1.35, transform=ax.transAxes)

    # Connecting arrows between planes
    _arrow(ax, 0.392, 0.67, 0.428, 0.67, color=BLUE, lw=2.0)
    _arrow(ax, 0.428, 0.31, 0.392, 0.31, color=EMERALD, lw=2.0)

    fig.savefig(OUT_DIR / "architecture-planes.png", bbox_inches="tight", pad_inches=0.18, facecolor=BG)
    plt.close(fig)


def draw_day10_poc_loop():
    fig, ax = _new_canvas(14.5, 5.8)
    ax.text(0.03, 0.92, "DAY 10 PROOF OF CONCEPT — CHU TRÌNH KIỂM ĐỊNH, TRUY VẾT & KHÔI PHỤC",
            fontsize=15.5, fontweight="bold", color=TEXT_DARK, transform=ax.transAxes)
    ax.text(0.03, 0.865, "Chứng minh năng lực nền tảng: Dữ liệu có thể được kiểm tra, từ chối (HOLD), truy vết tác động và khôi phục bằng chứng thực nghiệm",
            fontsize=10.2, color=TEXT_MUTED, transform=ax.transAxes)

    steps = [
        ("1. LOAD", "Nạp Dataset Clean", "Đọc dataset mẫu, manifest,\ntaxonomy & tạo snapshot\nSHA-256 ban đầu.", BLUE, BLUE_SOFT),
        ("2. INJECT", "Tiêm lỗi có kiểm soát", "Chèn lỗi T1–T8 (sai nhãn,\nbox lệch, rò scene) vào\nbản sao working copy.", AMBER, AMBER_SOFT),
        ("3. VALIDATE", "Quality Gate → HOLD", "Validator phát hiện lỗi,\nchặn release tự động\nvới quyết định HOLD.", RED, RED_SOFT),
        ("4. TRACE", "Truy vết Blast Radius", "Xác định dataset, release\nvà training run bị ảnh\nhưởng qua đồ thị Lineage.", INDIGO, INDIGO_SOFT),
        ("5. RECOVER", "Time Machine Restore", "Khôi phục snapshot sạch,\nxác minh checksum & chạy\nvalidation lại → VALIDATED.", EMERALD, EMERALD_SOFT),
    ]

    margin = 0.03
    gap = 0.025
    w = (1.0 - 2 * margin - 4 * gap) / 5
    h = 0.48
    y0 = 0.28
    for i, (badge, title, desc, col, soft) in enumerate(steps):
        x0 = margin + i * (w + gap)
        _box(ax, x0, y0, w, h, fc=CARD_BG, ec=col, lw=1.8, r=0.015)
        _box(ax, x0 + 0.012, y0 + h - 0.09, w - 0.024, 0.07, fc=soft, ec="none", r=0.01)
        ax.text(x0 + w / 2, y0 + h - 0.055, badge, ha="center", va="center", fontsize=10, fontweight="bold", color=col, transform=ax.transAxes)
        ax.text(x0 + w / 2, y0 + h - 0.16, title, ha="center", va="center", fontsize=10.2, fontweight="bold", color=TEXT_DARK, transform=ax.transAxes)
        ax.text(x0 + w / 2, y0 + 0.12, desc, ha="center", va="center", fontsize=8.8, color=TEXT_MUTED, linespacing=1.45, transform=ax.transAxes)
        if i < 4:
            _arrow(ax, x0 + w + 0.002, y0 + h / 2, x0 + w + gap - 0.002, y0 + h / 2, color="#475569", lw=2.0)

    _box(ax, 0.03, 0.07, 0.94, 0.13, fc=EMERALD_SOFT, ec=EMERALD, lw=1.2, r=0.012)
    ax.text(0.50, 0.155, "BẰNG CHỨNG BENCHMARK THỰC TẾ (reports/FAULT_BENCHMARK.json):  40/40 Fault Runs phát hiện 100% (T1–T8)  •  False Hold: 0%  •  False Release: 0%",
            ha="center", va="center", fontsize=9.8, fontweight="bold", color=EMERALD, transform=ax.transAxes)
    ax.text(0.50, 0.105, "Chạy lệnh:  python -m release_control demo   hoặc   .\\scripts\\run_demo.ps1   để tái lập toàn bộ chu trình với bằng chứng JSON & SQLite",
            ha="center", va="center", fontsize=9.0, color=TEXT_DARK, transform=ax.transAxes)

    fig.savefig(OUT_DIR / "day10-poc-loop.png", bbox_inches="tight", pad_inches=0.18, facecolor=BG)
    plt.close(fig)


def draw_roadmap_phases():
    fig, ax = _new_canvas(14.5, 5.4)
    ax.text(0.03, 0.915, "LỘ TRÌNH PHÁT TRIỂN SẢN PHẨM — 5 GIAI ĐOẠN (SCOPE & ROADMAP)",
            fontsize=15.5, fontweight="bold", color=TEXT_DARK, transform=ax.transAxes)
    ax.text(0.03, 0.855, "Từ nền tảng kiểm định trong phòng lab (Day 10) đến dây chuyền sản xuất dữ liệu tự lái khép kín",
            fontsize=10.2, color=TEXT_MUTED, transform=ax.transAxes)

    phases = [
        ("PHASE 1", "Lab Foundation\n(Hiện tại — Day 10)", "Contract, manifest, validator\nT1–T8, fault lab, snapshot,\nrelease state & CLI/Dashboard.", EMERALD, EMERALD_SOFT, "ĐÃ TRIỂN KHAI"),
        ("PHASE 2", "Integrated\nFactory", "Kết nối pipeline pre-labeling,\nCVAT live sync, QC queue\n& bản ghi lineage thực tế.", BLUE, BLUE_SOFT, "TIẾP THEO"),
        ("PHASE 3", "Team\nOperations", "Phân công, reviewer workflow,\nteam metrics, audit, phân quyền\n& xử lý nhiều job song song.", INDIGO, INDIGO_SOFT, "KẾ HOẠCH"),
        ("PHASE 4", "Large-Scale\nBenchmark", "Đánh giá đa worker trên dữ liệu\nđại diện; đo throughput, RAM,\nchi phí & điểm nghẽn 2M frames.", AMBER, AMBER_SOFT, "MỞ RỘNG"),
        ("PHASE 5", "Model-in-the-Loop\nFeedback", "Liên kết kết quả đánh giá model\nvới lỗi dữ liệu & ưu tiên mẫu\ncho vòng annotation tiếp theo.", BLUE, BLUE_SOFT, "TẦM NHÌN"),
    ]

    margin = 0.03
    gap = 0.022
    w = (1.0 - 2 * margin - 4 * gap) / 5
    h = 0.56
    y0 = 0.16
    for i, (ph, title, desc, col, soft, status) in enumerate(phases):
        x0 = margin + i * (w + gap)
        _box(ax, x0, y0, w, h, fc=CARD_BG, ec=col, lw=1.8, r=0.015)
        _box(ax, x0 + 0.012, y0 + h - 0.09, w - 0.024, 0.07, fc=soft, ec="none", r=0.01)
        ax.text(x0 + w / 2, y0 + h - 0.055, ph, ha="center", va="center", fontsize=10, fontweight="bold", color=col, transform=ax.transAxes)
        ax.text(x0 + w / 2, y0 + h - 0.18, title, ha="center", va="center", fontsize=10.2, fontweight="bold", color=TEXT_DARK, transform=ax.transAxes)
        ax.text(x0 + w / 2, y0 + 0.19, desc, ha="center", va="center", fontsize=8.7, color=TEXT_MUTED, linespacing=1.45, transform=ax.transAxes)
        # Status pill at bottom
        _box(ax, x0 + 0.025, y0 + 0.03, w - 0.05, 0.058, fc=soft, ec=col, lw=1.0, r=0.01)
        ax.text(x0 + w / 2, y0 + 0.059, status, ha="center", va="center", fontsize=8.0, fontweight="bold", color=col, transform=ax.transAxes)
        if i < 4:
            _arrow(ax, x0 + w + 0.002, y0 + h / 2, x0 + w + gap - 0.002, y0 + h / 2, color="#64748B", lw=1.8)

    fig.savefig(OUT_DIR / "roadmap-phases.png", bbox_inches="tight", pad_inches=0.18, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    draw_pipeline_7_stages()
    draw_scale_workload_split()
    draw_architecture_planes()
    draw_day10_poc_loop()
    draw_roadmap_phases()
    print("Generated 5 diagrams in", OUT_DIR)
