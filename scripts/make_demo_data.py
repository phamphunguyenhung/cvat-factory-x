"""Idempotent thin adapter: copy REAL lab clean delivery -> demo/coco128_clean and write
release_control-format manifest.json (lab manifest/metadata values kept verbatim: sha256, scene_id, split).
Source (read-only): labs/K5-L23-Day10-Dataset-Release-Handover/runs/clean/delivery
(images originate from data_source/coco250 val2017 subset)."""
import json, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "labs/K5-L23-Day10-Dataset-Release-Handover/runs/clean/delivery"
OUT = ROOT / "demo/coco128_clean"

def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(SRC, OUT)
    shutil.move(OUT / "manifest.json", OUT / "manifest.lab.json")
    lab = json.loads((OUT / "manifest.lab.json").read_text(encoding="utf-8"))
    meta = {m["image"]: m for m in json.loads((OUT / "metadata.json").read_text(encoding="utf-8"))["items"]}
    classes = [l.split(" ", 1)[1].strip() if l.split(" ", 1)[0].isdigit() else l.strip()
               for l in (OUT / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    entries = []
    for iid, it in lab["items"].items():
        m = meta[iid]
        entries.append({"image_id": iid, "image_path": it["image"], "label_path": it["label"],
                        "sha256": it["image_sha256"], "label_sha256": it["label_sha256"],
                        "mask_path": it["mask"], "mask_sha256": it["mask_sha256"],
                        "scene_id": m["scene_id"], "split": m["split"], "job_id": m["job_id"],
                        "guideline_version": m["guideline_version"]})
    gv = sorted({e["guideline_version"] for e in entries})
    manifest = {"dataset_id": lab["dataset"], "guideline_version": gv[-1], "classes": classes,
                "num_classes": lab["num_classes"], "license": "CC BY 4.0 (COCO)", "images": entries}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (ROOT / "demo/LICENSE_NOTE.md").write_text(
        "# License note\n\nImages/annotations derived from MS COCO val2017 (https://cocodataset.org), "
        "annotations CC BY 4.0; images carry Flickr licenses per COCO terms. Attribution: Lin et al., "
        "Microsoft COCO: Common Objects in Context.\nCopied unmodified from lab runs/clean/delivery; "
        "scene/job/annotator metadata is synthetic per the lab.\n", encoding="utf-8")
    print(f"{len(entries)} items, guideline versions {gv}")

main()
