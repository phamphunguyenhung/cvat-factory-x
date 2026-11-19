import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
import metrics  # noqa: E402

HAVE_DATA = (ROOT / "data_source" / "coco_panoptic250" / "panoptic.json").is_file()
HAVE_TEST = (ROOT / "data_source" / "coco_panoptic250_test" / "panoptic.json").is_file()


def run(script: str, *args: str, check=True) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPTS / script), *args], check=check, cwd=ROOT, capture_output=True, text=True)


class MetricsTest(unittest.TestCase):
    def test_empty_truth_and_prediction_is_perfect(self):
        row = metrics.score(set(), set())
        self.assertEqual((row["precision"], row["recall"]), (1.0, 1.0))

    def test_false_alarms_lower_f1(self):
        truth = {("class_swap", "a"), ("wrong_hash", "b")}
        self.assertEqual(metrics.score(truth, truth)["f1"], 1.0)
        noisy = truth | {("bbox_drift", f"x{i}") for i in range(6)}
        self.assertLess(metrics.score(truth, noisy)["f1"], 0.5)

    def test_groups_cover_all_types(self):
        self.assertEqual(metrics.RULE_TYPES & metrics.SEMANTIC_TYPES, set())
        self.assertIn("missing_box", metrics.SEMANTIC_TYPES)
        self.assertEqual(metrics.decision([{"type": "class_swap", "target": "a"}]), "HOLD")
        self.assertEqual(metrics.decision([]), "RELEASE_CANDIDATE")


@unittest.skipUnless(HAVE_DATA, "chưa tải dữ liệu: python scripts/download_public.py")
class ReleaseTest(unittest.TestCase):
    def test_same_seed_same_delivery_and_perfect_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            run("inject_release.py", "--seed", "11", "--out", str(a))
            run("inject_release.py", "--seed", "11", "--out", str(b))
            ra = json.loads((a / "delivery" / "release.json").read_text(encoding="utf-8"))["release_id"]
            rb = json.loads((b / "delivery" / "release.json").read_text(encoding="utf-8"))["release_id"]
            self.assertEqual(ra, rb)
            truth = json.loads((a / "_private" / "truth.json").read_text(encoding="utf-8"))
            kinds = {e["type"] for e in truth["errors"]}
            self.assertTrue(kinds & metrics.SEMANTIC_TYPES and kinds & metrics.RULE_TYPES)
            self.assertLessEqual(kinds, metrics.HOLD_TYPES)
            # mỗi ảnh nhiều nhất một lỗi
            targets = [e.get("image", e["target"]) for e in truth["errors"]]
            self.assertEqual(len(targets), len(set(targets)))
            (a / "errors.json").write_text(json.dumps({"errors": truth["errors"]}), encoding="utf-8")
            shutil.copyfile(a / "_private" / "trace_truth.json", a / "trace.json")
            run("score_findings.py", "--run", str(a), "--truth-dir", str(a / "_private"))
            report = json.loads((a / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["score"]["overall"]["f1"], 1.0)
            self.assertEqual(report["trace"]["tier_b"]["f1"], 1.0)
            # check_handover báo FAIL (chưa làm bàn giao) nhưng không lỗi chương trình
            out = run("check_handover.py", "--run", str(a), check=False)
            self.assertEqual(out.returncode, 1)
            self.assertIn("FAIL", out.stdout)

    @unittest.skipUnless((ROOT / "data_source" / "coco_panoptic250_calib" / "panoptic.json").is_file(), "chưa có pool calib")
    def test_clean_delivery_has_no_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = Path(tmp) / "c"
            run("inject_release.py", "--seed", "5", "--no-faults", "--out", str(c))
            self.assertEqual(json.loads((c / "_private" / "truth.json").read_text(encoding="utf-8"))["errors"], [])
            # lô sạch không dùng ảnh luyện tập (diff sẽ lộ đáp án)
            practice = {p.name for p in (ROOT / "data_source" / "coco250" / "images" / "train2017").glob("*.jpg")}
            self.assertFalse(practice & {p.name for p in (c / "delivery" / "images").glob("*.jpg")})


@unittest.skipUnless(HAVE_TEST, "chưa có pool test (giảng viên): python scripts/download_public.py --pool test")
class HiddenTest(unittest.TestCase):
    def test_opaque_delivery_does_not_reuse_coco_bytes(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            h = Path(tmp) / "h"
            run("inject_release.py", "--seed", "1234", "--pool", "test", "--opaque", "--out", str(h))
            src = ROOT / "data_source"
            coco = {hashlib.sha256(p.read_bytes()).hexdigest() for p in (src / "coco250_test" / "images" / "train2017").glob("*.jpg")}
            coco |= {hashlib.sha256(p.read_bytes()).hexdigest() for p in (src / "coco_panoptic250_test" / "masks").glob("*.png")}
            files = [p for p in (h / "delivery").rglob("*") if p.suffix in (".jpg", ".png")]
            self.assertTrue(files)
            self.assertFalse(any(hashlib.sha256(p.read_bytes()).hexdigest() in coco for p in files))
            self.assertTrue(all(p.stem.startswith("img_") for p in files))


if __name__ == "__main__":
    unittest.main()
