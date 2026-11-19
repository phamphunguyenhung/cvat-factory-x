"""Sinh hình minh hoạ (sơ đồ) cho README. Giảng viên dùng; học viên không cần chạy.

    python docs/assets/make_figures.py      # cần Pillow

Không dùng ảnh COCO trong hình (giấy phép ảnh COCO không đồng nhất).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONTS = ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"]
FONTS_B = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"]
NAVY, TEAL, ORANGE, GREEN, RED, PURPLE, GRAY, LIGHT = "#0b1f3a", "#12b3a8", "#ff8a3d", "#2fbf71", "#e5484d", "#7b61ff", "#6b7686", "#f3f6fa"


def font(size, bold=False):
    for p in (FONTS_B if bold else FONTS):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def center(d, xy, text, f, fill="white"):
    d.text((xy[0] - d.textlength(text, font=f) / 2, xy[1]), text, font=f, fill=fill)


def arrow(d, x0, x1, y, col=GRAY):
    d.line([(x0, y), (x1 - 14, y)], fill=col, width=5)
    d.polygon([(x1, y), (x1 - 18, y - 11), (x1 - 18, y + 11)], fill=col)


def hero():
    W, H = 1600, 420
    im = Image.new("RGB", (W, H), NAVY)
    d = ImageDraw.Draw(im)
    d.text((60, 70), "Day 10 Lab", font=font(34), fill=TEAL)
    d.text((60, 120), "Bàn giao dataset", font=font(64, True), fill="white")
    d.text((60, 200), "& Truy vết", font=font(64, True), fill=ORANGE)
    d.text((60, 300), "Bắt lỗi khi không có đáp án, bàn giao v2, chỉ ra ai cần sửa.", font=font(26), fill="#c9d4e2")
    # minh hoạ: khung ảnh, box detection lệch khỏi mask panoptic
    x0, y0 = 1010, 70
    d.rounded_rectangle((x0, y0, x0 + 500, y0 + 280), 18, fill="#16304f")
    d.ellipse((x0 + 90, y0 + 70, x0 + 260, y0 + 230), fill="#2f6f8f")
    d.ellipse((x0 + 290, y0 + 110, x0 + 420, y0 + 240), fill="#6b4f9f")
    d.rectangle((x0 + 90, y0 + 70, x0 + 260, y0 + 230), outline=GREEN, width=5)
    d.rectangle((x0 + 325, y0 + 95, x0 + 455, y0 + 225), outline=RED, width=5)
    d.text((x0 + 92, y0 + 36), "car: khớp mask", font=font(24, True), fill=GREEN)
    d.text((x0 + 300, y0 + 36), "truck? lệch", font=font(24, True), fill=RED)
    im.save(HERE / "hero.png")


def roadmap():
    steps = [("P1", "Bắt lỗi luật", "validator đọc delivery", GREEN), ("P2", "Đối chiếu", "det ↔ panoptic, xem ảnh", PURPLE),
             ("P3", "Quyết định", "ngưỡng 2%, lineage", ORANGE), ("P4", "Bàn giao v2", "không ghi đè v1", RED),
             ("P5", "Truy vết ticket", "tầng a/b, báo ai", TEAL), ("P6", "Bảo mật + đề ẩn", "40 phút cuối", NAVY)]
    W, H = 1600, 300
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    bw, gap = 215, 50
    x0 = (W - (6 * bw + 5 * gap)) // 2
    for i, (a, b, c, col) in enumerate(steps):
        x = x0 + i * (bw + gap)
        d.rounded_rectangle((x, 70, x + bw, 230), 18, fill=col)
        center(d, (x + bw / 2, 85), a, font(30, True))
        center(d, (x + bw / 2, 135), b, font(23, True))
        center(d, (x + bw / 2, 172), c, font(18))
        if i < 5:
            d.polygon([(x + bw + gap - 8, 150), (x + bw + 8, 133), (x + bw + 8, 167)], fill=GRAY)
    im.save(HERE / "roadmap.png")


def two_kinds():
    W, H = 1600, 520
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    cols = [(GREEN, "Lỗi luật", "nhìn file giao là thấy → script bắt được",
             ["missing_image · wrong_hash", "invalid_class · bbox_out_of_bounds", "duplicate_box · scene_leak",
              "guideline_mismatch", "invalid_category_id · thing_overlap"]),
            (RED, "Lỗi gán nhãn", "file vẫn hợp lệ, KHÔNG có nhãn gốc để so",
             ["class_swap · bbox_drift · missing_box", "category_swap · instance_merge", "",
              "→ đối chiếu detection ↔ panoptic ra ứng viên", "→ mở ảnh xem: lỗi thật hay lệch tự nhiên?"])]
    for i, (col, title, sub, rows) in enumerate(cols):
        x = 60 + i * 760
        d.rounded_rectangle((x, 40, x + 720, 480), 22, fill=LIGHT, outline=col, width=4)
        d.rounded_rectangle((x, 40, x + 720, 130), 22, fill=col)
        d.text((x + 30, 55), title, font=font(36, True), fill="white")
        d.text((x + 30, 150), sub, font=font(24, True), fill=NAVY)
        for k, r in enumerate(rows):
            d.text((x + 30, 210 + k * 50), r, font=font(24), fill=NAVY)
    im.save(HERE / "two_kinds.png")


def handover():
    W, H = 1600, 360
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    boxes = [(60, "Lô v1", "release_id gốc", "giữ nguyên, không sửa", NAVY), (440, "HOLD v1", "QC v1 > 2% ảnh lỗi", "lineage: decision", RED),
             (820, "Lô v2", "--parent v1", "manifest · summary · truyvet", GREEN), (1200, "R-021", "run đã train trên v1", "affected → review/retrain", ORANGE)]
    for x, a, b, c, col in boxes:
        d.rounded_rectangle((x, 80, x + 340, 280), 20, fill=col)
        center(d, (x + 170, 105), a, font(34, True))
        center(d, (x + 170, 165), b, font(22))
        center(d, (x + 170, 205), c, font(20))
    for x in (400, 780, 1160):
        arrow(d, x + 2, x + 40, 180)
    d.text((820, 300), "rework_queue.csv → ảnh lỗi giao lại đúng người gán/kiểm (jobs.csv)", font=font(20), fill=GRAY)
    im.save(HERE / "handover.png")


if __name__ == "__main__":
    hero(), roadmap(), two_kinds(), handover()
    print("ok")
