"""Interactive HTML Review Dashboard generator and decision persistence."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .models import ReviewBatch, ReviewCase


def _rel(target: str, base_dir: Path) -> str:
    try:
        return Path(target).resolve().relative_to(base_dir.resolve()).as_posix()
    except Exception:
        return Path(target).name


def render_review_dashboard(batch: ReviewBatch, out_html: Path) -> Path:
    """Write a self-contained Light Mode interactive HTML review dashboard."""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    base_dir = out_html.parent

    cases_payload = []
    cards_html = []
    for idx, c in enumerate(batch.cases, start=1):
        rel_before = _rel(c.before_path, base_dir)
        rel_after = _rel(c.after_path, base_dir)
        rel_overlay = _rel(c.overlay_path, base_dir)
        rel_zoom = _rel(c.crop_zoom_path, base_dir)
        rel_orig = _rel(c.original_crop_path, base_dir)
        rel_enh = _rel(c.enhanced_crop_path, base_dir)

        cases_payload.append(
            {
                "case_id": c.case_id,
                "frame_id": c.frame_id,
                "label": c.label,
                "priority": c.priority,
                "affected_points": c.affected_points,
                "decision": c.decision,
                "comment": c.comment,
                "before": rel_before,
                "after": rel_after,
                "overlay": rel_overlay,
                "zoom": rel_zoom,
                "original": rel_orig,
                "enhanced": rel_enh,
            }
        )

        issue_rows = "".join(
            f"<li><code>{html.escape(str(i.get('code', 'ISSUE')))}</code> "
            f"<strong>[{html.escape(str(i.get('region', '')))}]</strong>: "
            f"{html.escape(str(i.get('message', '')))}</li>"
            for i in c.issues
        ) or f"<li>{html.escape(c.reason or 'Evidence attached')}</li>"

        aff_str = ", ".join(str(p) for p in c.affected_points) if c.affected_points else "none"
        cid_esc = html.escape(c.case_id)

        cards_html.append(
            f"""
            <section class="card" data-case="{cid_esc}">
              <header class="card-head">
                <div>
                  <span class="badge">#{idx} Priority {c.priority:.2f}</span>
                  <strong>Frame {html.escape(str(c.frame_id))}</strong>
                  <span class="meta">Label: <code>{html.escape(c.label)}</code></span>
                  <span class="meta">Affected Points: <code>[{html.escape(aff_str)}]</code></span>
                </div>
                <div class="decision-bar" data-case="{cid_esc}">
                  <button type="button" class="btn btn-approve" onclick="setDecision('{cid_esc}','approve')">Approve</button>
                  <button type="button" class="btn btn-reject" onclick="setDecision('{cid_esc}','reject')">Reject</button>
                  <button type="button" class="btn btn-edit" onclick="setDecision('{cid_esc}','edit')">Needs Edit</button>
                  <span class="status-pill" id="status-{cid_esc}">{html.escape(c.decision)}</span>
                </div>
              </header>

              <div class="grid">
                <div class="panel">
                  <div class="toolbar">
                    <span>Main View:</span>
                    <button type="button" class="tab active" onclick="switchMain('{cid_esc}','overlay',this)">OVERLAY</button>
                    <button type="button" class="tab" onclick="switchMain('{cid_esc}','before',this)">BEFORE</button>
                    <button type="button" class="tab" onclick="switchMain('{cid_esc}','after',this)">AFTER</button>
                  </div>
                  <img id="main-img-{cid_esc}" src="{html.escape(rel_overlay)}" alt="Overlay {cid_esc}" />
                </div>

                <div class="panel">
                  <div class="toolbar">
                    <span>Error Region Crop:</span>
                    <button type="button" class="tab active" onclick="switchCrop('{cid_esc}','zoom',this)">Crop Zoom</button>
                    <button type="button" class="tab" onclick="switchCrop('{cid_esc}','original',this)">Original</button>
                    <button type="button" class="tab" onclick="switchCrop('{cid_esc}','enhanced',this)">Enhanced</button>
                  </div>
                  <img id="crop-img-{cid_esc}" src="{html.escape(rel_zoom)}" alt="Crop Zoom {cid_esc}" />
                </div>
              </div>

              <div class="evidence">
                <strong>Evidence &amp; Checks:</strong>
                <ul>{issue_rows}</ul>
                <label class="comment-row">
                  Note:
                  <input type="text" id="comment-{cid_esc}" value="{html.escape(c.comment)}"
                         placeholder="Optional reviewer note..." oninput="setComment('{cid_esc}', this.value)" />
                </label>
              </div>
            </section>
            """
        )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CVAT Factory Review — Batch {html.escape(batch.batch_id)}</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --surface: #ffffff;
      --border: #cbd5e1;
      --text: #0f172a;
      --muted: #475569;
      --primary: #1d4ed8;
      --ok: #15803d;
      --warn: #b45309;
      --bad: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      font-size: 14px;
      line-height: 1.5;
    }}
    .page {{ width: 100%; padding: 20px 28px; }}
    .topbar {{
      display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
      gap: 12px; padding: 14px 18px; background: var(--surface);
      border: 1px solid var(--border); border-radius: 8px; margin-bottom: 18px;
    }}
    .topbar h1 {{ margin: 0; font-size: 20px; }}
    .card {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 8px; padding: 16px; margin-bottom: 18px;
    }}
    .card-head {{
      display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
      gap: 10px; padding-bottom: 12px; border-bottom: 1px solid var(--border); margin-bottom: 12px;
    }}
    .badge {{
      display: inline-block; padding: 2px 8px; border-radius: 4px;
      background: #dbeafe; color: #1e3a8a; font-weight: 600; margin-right: 8px;
    }}
    .meta {{ margin-left: 10px; color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    .panel {{ border: 1px solid var(--border); border-radius: 6px; padding: 10px; background: #f1f5f9; }}
    .panel img {{ width: 100%; height: auto; display: block; border-radius: 4px; background: #ffffff; }}
    .toolbar {{ display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-bottom: 8px; font-weight: 600; }}
    .tab, .btn {{
      height: 36px; padding: 0 12px; border-radius: 6px; border: 1px solid var(--border);
      background: var(--surface); color: var(--text); cursor: pointer; font-weight: 600;
    }}
    .tab.active {{ background: var(--primary); color: #ffffff; border-color: var(--primary); }}
    .btn-approve {{ border-color: var(--ok); color: var(--ok); }}
    .btn-reject {{ border-color: var(--bad); color: var(--bad); }}
    .btn-edit {{ border-color: var(--warn); color: var(--warn); }}
    .status-pill {{
      display: inline-block; padding: 4px 10px; border-radius: 4px;
      background: #e2e8f0; font-weight: 700; text-transform: uppercase; font-size: 12px;
    }}
    .evidence {{ margin-top: 12px; }}
    .evidence ul {{ margin: 6px 0 10px 18px; padding: 0; }}
    .comment-row {{ display: flex; gap: 8px; align-items: center; }}
    .comment-row input {{
      flex: 1; height: 36px; padding: 0 10px; border: 1px solid var(--border); border-radius: 6px;
    }}
  </style>
</head>
<body>
  <main class="page">
    <div class="topbar">
      <div>
        <h1>CVAT Factory Automated Review — Batch {html.escape(batch.batch_id)}</h1>
        <div>Job: <strong>{html.escape(str(batch.job_id or 'N/A'))}</strong> &middot; Showing {len(batch.cases)} highest-priority case(s) (max 5)</div>
      </div>
      <div>
        <button type="button" class="tab active" onclick="exportDecisions()">Export decisions.json</button>
      </div>
    </div>
    {''.join(cards_html)}
  </main>
  <script>
    const CASES = {json.dumps(cases_payload)};
    const STORAGE_KEY = "cvat_factory_review_" + {json.dumps(batch.batch_id)};
    const state = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{{}}");

    function findCase(id) {{ return CASES.find(c => c.case_id === id); }}

    function switchMain(id, mode, btn) {{
      const c = findCase(id); if (!c) return;
      document.getElementById("main-img-" + id).src = c[mode];
      btn.parentElement.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
    }}

    function switchCrop(id, mode, btn) {{
      const c = findCase(id); if (!c) return;
      document.getElementById("crop-img-" + id).src = c[mode];
      btn.parentElement.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
    }}

    function setDecision(id, decision) {{
      state[id] = state[id] || {{}};
      state[id].decision = decision;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
      const pill = document.getElementById("status-" + id);
      if (pill) pill.textContent = decision;
    }}

    function setComment(id, comment) {{
      state[id] = state[id] || {{}};
      state[id].comment = comment;
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    }}

    function exportDecisions() {{
      const out = CASES.map(c => ({{
        case_id: c.case_id,
        frame_id: c.frame_id,
        label: c.label,
        decision: (state[c.case_id] && state[c.case_id].decision) || c.decision,
        comment: (state[c.case_id] && state[c.case_id].comment) || c.comment || ""
      }}));
      const blob = new Blob([JSON.stringify(out, null, 2)], {{type: "application/json"}});
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "decisions.json";
      a.click();
    }}
  </script>
</body>
</html>
"""
    out_html.write_text(doc, encoding="utf-8")
    return out_html


def save_decisions(decisions_path: Path, cases: list[ReviewCase]) -> Path:
    """Persist current case decisions to JSON on disk."""
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "case_id": c.case_id,
            "frame_id": c.frame_id,
            "label": c.label,
            "priority": c.priority,
            "affected_points": c.affected_points,
            "decision": c.decision,
            "comment": c.comment,
        }
        for c in cases
    ]
    decisions_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return decisions_path


def record_decision(
    decisions_path: Path,
    case_id: str,
    decision: str,
    comment: str = "",
) -> list[dict[str, Any]]:
    """Update a user decision for a given case_id in decisions.json."""
    if decision not in {"pending", "approve", "reject", "edit"}:
        raise ValueError(f"Invalid decision: {decision}")
    data = json.loads(decisions_path.read_text(encoding="utf-8")) if decisions_path.exists() else []
    updated = False
    for row in data:
        if row.get("case_id") == case_id:
            row["decision"] = decision
            row["comment"] = comment
            updated = True
            break
    if not updated:
        data.append({"case_id": case_id, "decision": decision, "comment": comment})
    decisions_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data
