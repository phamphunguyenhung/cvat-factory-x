"""VLM/LLM phan xu NGU NGHIA cho cac FLAG_REVIEW. Mac dinh TAT (ruleset decision.vlm.enabled=false).

LLM chi duoc tra ACTION tham chieu id co san (action_schema); khong toa do, khong tu dat id. Action sai schema bi bo,
shape giu FLAG_REVIEW. Backend:
  claude_session : ghi 'adjudication pack' (crop + issues + action cho phep + id hop le) -> nguoi/phien Claude viet
                   file JSON action -> nap lai + validate nghiem.
  ollama         : HTTP localhost:11434 (khong tai model; server tat / thieu model -> FLAG_REVIEW, khong loi).
"""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path

from cvat_tools.decision import action_schema as S


def enabled(rs: dict) -> bool:
    return bool((((rs.get("decision") or {}).get("vlm")) or {}).get("enabled", False))


def make_adjudicator(rs: dict, out_dir=None, response_file=None, img=None):
    """None neu tat. Backend theo rs['decision']['vlm']['backend']."""
    if not enabled(rs):
        return None
    v = rs["decision"]["vlm"]
    if v.get("backend") == "ollama":
        return OllamaAdjudicator(v.get("ollama") or {}, img)
    return ClaudeSessionAdjudicator(out_dir or Path("adjudication"), response_file)


class Adjudicator:
    """adjudicate(ctx, flagged_records) -> {tuple(shape_ids): record_moi}. Record chi duoc tra neu qua validate_record."""
    name = "base"
    model_id: str | None = None
    model_version: str | None = None

    def propose(self, ctx: dict, flagged: list[dict]) -> list[dict]:
        raise NotImplementedError

    def adjudicate(self, ctx: dict, flagged: list[dict]) -> dict:
        out = {}
        for act in self.propose(ctx, flagged):
            errs = S.validate_action(act, ctx)
            if errs or act.get("action") == "FLAG_REVIEW":
                continue
            rec = S.new_record(ctx["job"], ctx["frame"], act, model_id=self.model_id or self.name, model_version=self.model_version)
            rec["calibrated"] = False
            rec["priority_score"] = next((f["priority_score"] for f in flagged if f["shape_ids"][0] in act["shape_ids"]), None)
            if not S.validate_record(rec, ctx):
                out[tuple(act["shape_ids"])] = rec
        return out


def pack_items(ctx: dict, flagged: list[dict]) -> dict:
    """Noi dung pack: chi ID/nhan/issue da co; khong day toa do cho LLM chon."""
    return {
        "schema_version": S.SCHEMA_VERSION, "job": ctx["job"], "frame": ctx["frame"],
        "allowed_actions": [a for a in S.ACTIONS if a != "KEEP"] + ["KEEP"],
        "valid_ids": {"shape_ids": sorted(ctx["shapes"]), "roi_ids": sorted(ctx["rois"]), "proposal_ids": sorted(ctx["proposals"])},
        "action_format": {"schema_version": 1, "action": "...", "shape_ids": ["<id co san>"], "params": "xem params_by_action",
                          "reason_code": "CHU_HOA", "evidence_refs": ["issue:<shape>:<CODE>"], "confidence": "0..1 hoac null"},
        "params_by_action": {k: sorted(v) for k, v in S.PARAM_KEYS.items()},
        "rules": ["KHONG ghi toa do/polygon", "KHONG tu dat id", "khong du bang chung -> FLAG_REVIEW",
                  "shape origin=human chi KEEP/FLAG_REVIEW"],
        "items": [{"shape_ids": f["shape_ids"], "labels": [ctx["shapes"][i]["label"] for i in f["shape_ids"]],
                   "origin": [S.origin_of(ctx["shapes"][i]) for i in f["shape_ids"]], "reason_code": f["reason_code"],
                   "evidence_refs": f["evidence_refs"], "note": f["params"].get("note")} for f in flagged],
    }


class ClaudeSessionAdjudicator(Adjudicator):
    name = "claude_session"

    def __init__(self, out_dir, response_file=None, img=None):
        self.out_dir, self.response_file, self.img = Path(out_dir), (Path(response_file) if response_file else None), img
        self.model_id, self.model_version = "claude_session", None

    def write_pack(self, ctx: dict, flagged: list[dict]) -> Path:
        """Ghi pack: adjudication_<frame>.json + crop quanh tung shape (neu co anh)."""
        from cvat_tools.review import pack as P
        from cvat_tools.review import validators as V
        d = self.out_dir / f"frame_{ctx['frame']}"
        d.mkdir(parents=True, exist_ok=True)
        doc = pack_items(ctx, flagged)
        if self.img is not None:
            import cv2
            for it in doc["items"]:
                sh = ctx["shapes"][it["shape_ids"][0]]
                x0, y0, x1, y1 = V.poly_of(sh).bounds
                m = max(30.0, 0.35 * max(x1 - x0, y1 - y0))
                im, meta = P.render_crop(self.img, sh, (x0 - m, y0 - m, x1 + m, y1 + m))
                fn = f"crop_{sh['id']}.jpg"
                cv2.imwrite(str(d / fn), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
                it["crop"] = {"file": fn, **meta}
        f = d / "adjudication_pack.json"
        f.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf8")
        return f

    def propose(self, ctx, flagged):
        self.write_pack(ctx, flagged)
        f = self.response_file or (self.out_dir / f"frame_{ctx['frame']}" / "actions.json")
        if not f.exists():
            return []
        return load_actions_file(f)


def load_actions_file(path) -> list[dict]:
    """Doc file JSON action (list, hoac {'actions': [...]}). File hong -> []; tung action van duoc validate sau do."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf8"))
    except (OSError, ValueError):
        return []
    acts = doc.get("actions") if isinstance(doc, dict) else doc
    return [a for a in acts if isinstance(a, dict)] if isinstance(acts, list) else []


class OllamaAdjudicator(Adjudicator):
    name = "ollama"

    def __init__(self, cfg: dict, img=None):
        self.url = (cfg.get("url") or "http://localhost:11434").rstrip("/")
        self.model = cfg.get("model") or ""
        self.timeout = float(cfg.get("timeout_s", 20))
        self.img = img
        self.model_id, self.model_version = f"ollama:{self.model}", None
        self.last_error: str | None = None

    def _get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf8"))

    def available(self) -> bool:
        """Server song VA model da co san (chi GET /api/tags; khong bao gio pull)."""
        if not self.model:
            self.last_error = "chua cau hinh model"
            return False
        try:
            tags = self._get("/api/tags")
        except (urllib.error.URLError, OSError, ValueError) as e:
            self.last_error = f"ollama khong chay: {e}"
            return False
        names = {m.get("name") for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            self.last_error = f"model {self.model} chua co trong ollama (khong tu tai)"
            return False
        return True

    def propose(self, ctx, flagged):
        if not self.available():
            return []
        out = []
        for f in flagged:
            doc = pack_items(ctx, [f])
            msg = {"role": "user", "content": "Tra DUY NHAT mot JSON array cac action theo action_format, chi dung id trong valid_ids. "
                   + json.dumps(doc, ensure_ascii=False)}
            if self.img is not None:
                import cv2
                ok, buf = cv2.imencode(".jpg", self.img)
                if ok:
                    msg["images"] = [base64.b64encode(buf.tobytes()).decode()]
            body = json.dumps({"model": self.model, "messages": [msg], "stream": False, "format": "json"}).encode()
            try:
                req = urllib.request.Request(self.url + "/api/chat", body, {"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    txt = json.loads(r.read().decode("utf8"))["message"]["content"]
                data = json.loads(txt)
            except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
                self.last_error = f"ollama loi: {e}"
                continue
            acts = data.get("actions") if isinstance(data, dict) else data
            out += [a for a in acts if isinstance(a, dict)] if isinstance(acts, list) else []
        return out
