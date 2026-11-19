"""Tao/doc/xoa session demo tren CVAT LOCAL va client ghi bi write-guard.

Chi cho phep host localhost/127.0.0.1. Khong bao gio dung CVAT_HOST/CVAT_PAT (thuong tro toi edufun).
Credential: CVAT_LOCAL_TOKEN | CVAT_LOCAL_USER+CVAT_LOCAL_PASSWORD (env/.env), fallback cvat/config/cvat.local.yaml.
"""
from __future__ import annotations

import base64
import colorsys
import json
import os
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests

from release_control.config import guard_cvat_write, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SESSION_PATH = PROJECT_ROOT / "artifacts" / "release_control_demo" / "session.json"
IMAGES_DIR = PROJECT_ROOT / "demo" / "coco128_clean" / "images"
TAXONOMY = PROJECT_ROOT / "release_control" / "contracts" / "taxonomy.yaml"
CVAT_LOCAL_YAML = PROJECT_ROOT.parents[1] / "cvat" / "config" / "cvat.local.yaml"
LOCAL_HOSTS = {"localhost", "127.0.0.1"}
DEFAULT_HOST = "http://localhost:8080"


class DemoSessionError(RuntimeError):
    pass


def assert_local(host: str) -> str:
    h = urlparse(host).hostname
    if h not in LOCAL_HOSTS:
        raise DemoSessionError(f"Tu choi: host '{h}' khong phai CVAT local (chi localhost/127.0.0.1).")
    return host.rstrip("/")


def _cvat_host() -> str:
    # Chi dung CVAT_LOCAL_HOST; CVAT_HOST bi bo qua vi co the tro toi edufun.
    load_dotenv()
    return assert_local(os.environ.get("CVAT_LOCAL_HOST") or DEFAULT_HOST)


def _auth_headers() -> dict:
    load_dotenv()
    tok = os.environ.get("CVAT_LOCAL_TOKEN") or os.environ.get("CVAT_LOCAL_PAT")
    user, pw = os.environ.get("CVAT_LOCAL_USER"), os.environ.get("CVAT_LOCAL_PASSWORD")
    if not (tok or (user and pw)) and CVAT_LOCAL_YAML.is_file():
        try:
            import yaml
            y = yaml.safe_load(CVAT_LOCAL_YAML.read_text(encoding="utf-8")) or {}
            tok = tok or y.get("token") or y.get("pat")
            user, pw = user or y.get("user") or y.get("username"), pw or y.get("password")
        except Exception:
            pass
    if tok:
        h = {"Authorization": f"Bearer {tok}"}
    elif user and pw:
        h = {"Authorization": "Basic " + base64.b64encode(f"{user}:{pw}".encode()).decode()}
    else:
        raise DemoSessionError(
            "Thieu credential CVAT LOCAL: dat CVAT_LOCAL_TOKEN hoac CVAT_LOCAL_USER + CVAT_LOCAL_PASSWORD "
            "(bien moi truong hoac day10/projects/.env).")
    h["Accept"] = "application/vnd.cvat+json"
    return h


def _api(method: str, host: str, path: str, headers: dict, **kw) -> requests.Response:
    assert_local(host)
    return requests.request(method, f"{host}{path}", headers=headers, timeout=kw.pop("timeout", 60), **kw)


def _check(r: requests.Response, what: str) -> requests.Response:
    if r.status_code >= 400:
        raise DemoSessionError(f"{what} loi HTTP {r.status_code}: {r.text[:300]}")
    return r


def _labels_from_taxonomy() -> list[dict]:
    import yaml
    tax = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    classes = tax["profiles"]["coco80"]["classes"]
    names = [classes[k] for k in sorted(classes)]
    out = []
    for i, n in enumerate(names):
        r, g, b = colorsys.hsv_to_rgb(i / len(names), 0.7, 0.85)
        out.append({"name": n, "color": "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)),
                    "type": "rectangle", "attributes": []})
    return out


def create_session(n_images: int = 20) -> dict:
    host = _cvat_host()
    h = _auth_headers()
    _check(_api("GET", host, "/api/users/self", h), "Dang nhap CVAT local")
    imgs = sorted(p for p in IMAGES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:n_images]
    if not imgs:
        raise DemoSessionError(f"Khong co anh trong {IMAGES_DIR}")
    name = "RC-DEMO-" + datetime.now().strftime("%Y%m%d-%H%M")
    task = _check(_api("POST", host, "/api/tasks", h, json={"name": name, "labels": _labels_from_taxonomy()}),
                  "Tao task").json()
    tid = task["id"]
    try:
        files = [(f"client_files[{i}]", (p.name, p.read_bytes(), "image/jpeg")) for i, p in enumerate(imgs)]
        data = {"image_quality": "95", "sorting_method": "lexicographical", "use_cache": "true"}
        rd = _check(_api("POST", host, f"/api/tasks/{tid}/data", h, data=data, files=files, timeout=300),
                    "Upload anh").json()
        rq = (rd or {}).get("rq_id")
        for _ in range(90):
            if not rq:
                time.sleep(3)
                break
            st = _check(_api("GET", host, f"/api/requests/{rq}", h), "Poll upload").json()
            if st.get("status") == "finished":
                break
            if st.get("status") == "failed":
                raise DemoSessionError(f"CVAT xu ly anh that bai: {st.get('message')}")
            time.sleep(2)
        else:
            raise DemoSessionError("Het gio cho CVAT xu ly anh.")
        jobs = _check(_api("GET", host, "/api/jobs", h, params={"task_id": tid}), "Lay job").json()["results"]
        if not jobs:
            raise DemoSessionError("Task khong co job.")
        labels = _check(_api("GET", host, "/api/labels", h, params={"task_id": tid, "page_size": 200}),
                        "Lay nhan").json()["results"]
    except Exception:
        _api("DELETE", host, f"/api/tasks/{tid}", h)  # don task do chinh minh vua tao
        raise
    session = {
        "cvat_host": host, "task_id": tid, "job_id": jobs[0]["id"], "task_name": name,
        "image_files": [p.name for p in imgs],
        "label_map": {l["name"]: l["id"] for l in labels},
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "created_by": "release_control",
    }
    SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SESSION_PATH.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8")
    return session


def load_session() -> dict:
    if not SESSION_PATH.is_file():
        raise DemoSessionError(f"Chua co session: {SESSION_PATH} (chay `ingest` truoc).")
    return json.loads(SESSION_PATH.read_text(encoding="utf-8"))


def job_url(session: dict) -> str:
    return f"{session['cvat_host']}/tasks/{session['task_id']}/jobs/{session['job_id']}"


def _guard(session: dict, action: str) -> None:
    guard_cvat_write(action)
    host = assert_local(session["cvat_host"])
    saved = load_session()  # nguon su that: session.json, khong tin dict truyen vao
    if str(session.get("task_id")) != str(saved.get("task_id")) or saved.get("created_by") != "release_control":
        raise DemoSessionError(f"Write-guard: task {session.get('task_id')} khong thuoc session.json")
    try:
        from release_control.audit import guard_write
    except ImportError:
        return
    guard_write(host, saved["task_id"], saved)


def cleanup_session() -> bool:
    session = load_session()
    _guard(session, "cleanup")
    h = _auth_headers()
    r = _api("DELETE", session["cvat_host"], f"/api/tasks/{session['task_id']}", h)
    if r.status_code not in (204, 404):
        _check(r, "Xoa task")
    SESSION_PATH.unlink(missing_ok=True)
    return True


class CvatDemoClient:
    def __init__(self, session: dict | None = None):
        self.session = session or load_session()
        self.host = assert_local(self.session["cvat_host"])
        self.job_id = self.session["job_id"]
        self._h = _auth_headers()

    def get_job_annotations(self) -> dict:
        return _check(_api("GET", self.host, f"/api/jobs/{self.job_id}/annotations", self._h),
                      "Doc annotation").json()

    def put_annotations(self, shapes: list[dict]) -> dict:
        _guard(self.session, "put_annotations")
        lm = self.session["label_map"]
        out = []
        for s in shapes:
            s = dict(s)
            if "label_id" not in s:
                s["label_id"] = lm[s.pop("label")]
            s.pop("label", None)
            for k, v in (("frame", 0), ("type", "rectangle"), ("occluded", False), ("z_order", 0),
                         ("rotation", 0), ("outside", False), ("attributes", []), ("group", 0),
                         ("source", "auto")):
                s.setdefault(k, v)
            out.append(s)
        body = {"version": 0, "tags": [], "shapes": out, "tracks": []}
        _check(_api("PUT", self.host, f"/api/jobs/{self.job_id}/annotations", self._h, json=body),
               "Ghi annotation")
        return self.get_job_annotations()
