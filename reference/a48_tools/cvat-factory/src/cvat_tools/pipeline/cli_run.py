"""`python -m cvat_tools run --job 285 --frames 71` / `--frames 60-79`. CHI DOC CVAT; ghi anh + bao cao + shapestore (rule_draft)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2

from cvat_tools.core import paths
from cvat_tools.pipeline import export as EX
from cvat_tools.pipeline import run as R
from cvat_tools.pipeline import targeteval as TE
from cvat_tools.pipeline.source import FrameSource
from cvat_tools.segmentation import labelmap as LM


def run_job(job: int, frames, out: Path, *, store_path=None, store: bool = True, use_sam: bool = False, tiles: bool = True,
            budget_s: float = 10.0, models=None, source=None, note: str = "") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    src = source or FrameSource(job)
    models = models or R.Models()
    st = None
    if store:
        from cvat_tools.core.shapestore import ShapeStore
        st = ShapeStore(store_path)
    run_id = R.new_run_id()
    results, tiles_img, cmp_by_frame, gaps = [], {}, {}, {}
    for f in frames:
        img = src.image(f)
        tgt = src.target(f)
        existing = tgt if tgt is not None else src.shapes(f)
        roads = [s["points"] for s in existing if s["label"] == "road" and s["type"] == "polygon"]
        human = [s for s in existing if s["label"] != "road"]
        r = R.process_frame(img, f, models, roads, human, job=job, use_sam=use_sam, tiles=tiles, budget_s=budget_s,
                            store=st, run_id=run_id)
        results.append(r)
        tool_img = EX.render_tool(img, r)
        cv2.imwrite(str(out / f"f{f}_tool.jpg"), tool_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        tiles_img[f] = tool_img
        if existing:
            cv2.imwrite(str(out / f"f{f}_side_by_side.jpg"),
                        EX.side_by_side(img, f, existing, r, "NGUOI VE (target)" if tgt is not None else "CVAT hien co"),
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
        H, W = img.shape[:2]
        gimg, gs = LM.render_gap(img, r["road_input"] + r["shapes"], f"f{f} TOOL")
        cv2.imwrite(str(out / f"f{f}_gapmap.jpg"), gimg, [cv2.IMWRITE_JPEG_QUALITY, 90])
        gaps[f] = {"tool_uncovered_pct": gs["uncovered_pct"]}
        if tgt is not None:
            gaps[f]["target_uncovered_pct"] = LM.gap_stats(tgt, W, H)["uncovered_pct"]
            cmp_by_frame[f] = TE.compare([s for s in tgt if s["label"] != "road"], r["shapes"])
        print(f"f{f}: {len(r['shapes'])} shape, {r['timings']['total']}s {r['timings']}", flush=True)
    if len(tiles_img) > 1:
        cv2.imwrite(str(out / "contact_sheet.jpg"), EX.contact_sheet(tiles_img), [cv2.IMWRITE_JPEG_QUALITY, 88])
    EX.write_csv(out / "run_report.csv", results)
    ft = next(iter(cmp_by_frame), None)
    notes = [f"run_id {run_id}; store {'ghi rule_draft' if st else 'KHONG ghi'}; sam {'bat' if use_sam else 'tat'}; tile {'bat' if tiles else 'tat'}"]
    if note:
        notes.append(note)
    notes.append("gap map: " + ", ".join(f"f{k}: tool {v['tool_uncovered_pct']}%" + (f" / target {v['target_uncovered_pct']}%" if "target_uncovered_pct" in v else "")
                                          for k, v in gaps.items()))
    EX.write_markdown(out / "run_report.md", results, job=job, budget_s=budget_s, models_load_s=getattr(models, "load_s", None),
                      target_cmp=cmp_by_frame.get(ft), target_weak=TE.weak(cmp_by_frame[ft]) if ft is not None else None,
                      notes=notes, frame_target=ft)
    (out / "run_result.json").write_text(json.dumps({"results": results, "gaps": gaps}, ensure_ascii=False), encoding="utf8")
    if st:
        st.close()
    return {"results": results, "gaps": gaps, "cmp": cmp_by_frame, "out": str(out)}


def cmd_run(a):
    out = Path(a.out) if a.out else paths.DATA / "jobs" / str(a.job) / "previews" / "run"
    r = run_job(a.job, R.parse_frames(a.frames), out, store=not a.no_store, use_sam=a.sam, tiles=not a.no_tiles, budget_s=a.budget)
    print("Da luu:", r["out"])
    return 0


def register(sub, add):
    s = add(sub, "run", cmd_run, help="Sinh nhap MOI nhan cho job (chi doc CVAT): anh + run_report + shapestore rule_draft")
    s.add_argument("--job", type=int, required=True)
    s.add_argument("--frames", required=True, help="vd 71 | 60-79 | 60,65,70")
    s.add_argument("--out", default=None)
    s.add_argument("--no-store", action="store_true", help="khong ghi shapestore")
    s.add_argument("--sam", action="store_true", help="MobileSAM cho vat the de len nen (neu con ngan sach)")
    s.add_argument("--no-tiles", action="store_true")
    s.add_argument("--budget", type=float, default=10.0)
