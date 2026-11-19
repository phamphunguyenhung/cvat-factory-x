"""Tang quyet dinh tren nhap job: luat tat dinh -> action -> ap dung len BAN SAO -> validate lai (so loi truoc/sau that).

  python scripts\\decide.py --job 285 --frames 61,63,69               # dry-run: kho trong bo nho, khong ghi gi
  python scripts\\decide.py --job 285 --frames 61 --write             # ghi kho shapes.sqlite (+decisions) va ban sua ra thu muc rieng
Khong ghi CVAT, khong ghi de nhap goc.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
from review_pack import DEFAULT_DRAFTS, FRAMES, load_frame_shapes  # noqa: E402

from cvat_tools.core.shapestore import ShapeStore  # noqa: E402
from cvat_tools.decision import engine as EN  # noqa: E402
from cvat_tools.decision import vlm_adjudicator as VA  # noqa: E402
from cvat_tools.review import apply as A  # noqa: E402
from cvat_tools.review import validators as V  # noqa: E402


def n_err(issues):
    return sum(1 for i in issues if i["severity"] == "error")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", type=int, default=285)
    ap.add_argument("--frames", default="61,63,69")
    ap.add_argument("--drafts", default=",".join(DEFAULT_DRAFTS))
    ap.add_argument("--frames-dir", default=str(FRAMES))
    ap.add_argument("--write", action="store_true", help="ghi kho SQLite + ban sua (mac dinh dry-run)")
    ap.add_argument("--db", default="", help="duong dan DB khi --write (mac dinh data/store/shapes.sqlite)")
    ap.add_argument("--out", default=str(ROOT / "data" / "jobs" / "285" / "previews" / "decide"))
    ap.add_argument("--rounds", type=int, default=2)
    a = ap.parse_args()
    rs = V.load_ruleset(a.job)
    names = [x for x in a.drafts.split(",") if x]
    store = ShapeStore(a.db or None) if a.write else ShapeStore(":memory:")
    table: Counter = Counter()
    tot_b: Counter = Counter()
    tot_a: Counter = Counter()
    eb = ea = 0
    for fr in [int(x) for x in a.frames.split(",")]:
        img = cv2.imread(str(Path(a.frames_dir) / f"j{a.job}_f{fr}.jpg"))
        if img is None:
            print(f"frame {fr}: thieu anh", file=sys.stderr)
            continue
        shapes = load_frame_shapes(names, fr, rs)
        adj = VA.make_adjudicator(rs, Path(a.out) / "adjudication", img=img)
        res = EN.run_frame(a.job, fr, shapes, img, rs, store, rounds=a.rounds, adjudicator=adj)
        for r in res["records"]:
            table[(r["reason_code"], r["proposed_action"], r["validation_status"])] += 1
        tot_b.update(res["counts_before"])
        tot_a.update(res["counts_after"])
        b, af = n_err(res["issues_before"]), n_err(res["issues_after"])
        eb, ea = eb + b, ea + af
        print(f"frame {fr}: {len(shapes)} shape -> {len(res['shapes'])}; loi error {b} -> {af}; tong loi {len(res['issues_before'])} -> {len(res['issues_after'])}")
        if a.write:
            saved = A.save_result({**res, "log": [], "remaining": {}, "touched_still_failing": {}, "resolved": [], "new_failures": []},
                                  Path(a.out) / "edited", tag=f"frame{fr}_decide")
            Path(saved["log"]).write_text(json.dumps({"records": res["records"]}, ensure_ascii=False, indent=1), encoding="utf8")
    print("\nBANG ACTION (reason_code | action | trang thai | so luong)")
    for (rc, ac, st), n in sorted(table.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {rc:<14} {ac:<14} {st:<17} {n}")
    codes = sorted(set(tot_b) | set(tot_a), key=lambda c: -tot_b.get(c, 0))
    print("\nLOI THEO MA (truoc -> sau)")
    for c in codes:
        print(f"  {c:<16} {tot_b.get(c, 0):>4} -> {tot_a.get(c, 0)}")
    print(f"\nTONG loi muc error: {eb} -> {ea}   (dry-run={not a.write}, vlm={'bat' if VA.enabled(rs) else 'tat'})")
    store.close()


if __name__ == "__main__":
    main()
