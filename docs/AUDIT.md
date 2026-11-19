# AUDIT codebase release_control (doi chieu de bai)

Pham vi: `release_control/` (cli, config, adapters, release/gate, versions/restore, lineage, dashboard/app). Chi doc, khong sua code.
Module moi: `release_control/audit/__init__.py` (AuditLog hash-chain, guard_write, redact) + `tests/test_audit.py` (5 test pass).

## Kiem tra nhanh
| Muc | Ket qua |
|---|---|
| Ro ri secret | Quet gia tri CVAT_PAT / CVAT_EDUFUN_PAT / CVAT_A48_TOKEN cua `.env` trong toan bo repo (ke ca reports, artifacts, sqlite): **0 khop**. `.env` nam trong `.gitignore`, chua duoc track. |
| Release thieu approval | `gate.approve_release` bat buoc qc+privacy APPROVED (gate.py:203-217); `publish` yeu cau APPROVED; `accept` yeu cau PUBLISHED + consumer. OK. |
| PUBLISHED != ACCEPTED | Tach trang thai dung (gate.py:222-281, policy `published_is_not_accepted`). OK. |
| Evidence thuc te | `reports/RELEASE_EVIDENCE.json`: status VALIDATED, approvals=[] , consumer_acks=[] (khong bia). OK. |
| pytest | `tests/test_audit.py` 5 pass. `tests/test_cli.py::test_demo_reaches_validated_without_approval` loi PermissionError WinError 32 khi `shutil.rmtree` file `artifacts/release_control_demo/mlruns/mlflow.db` dang bi tien trinh khac giu (xung dot Windows/dung chung thu muc artifacts khi nhieu agent chay); chua chay het bo test vi dung chung artifacts. |

## Bang phat hien
| # | Muc do | File:dong | Mo ta | De xuat |
|---|---|---|---|---|
| 1 | CAO | dashboard/app.py:331-348 | Bootstrap ghi canh `release -trained_in-> training_run` va `produces_model` voi `evidence_status="verified"` du la `run_origin="simulated_demo"` (khong co training that). Vi pham "lineage chi canh co bang chung". | Dat `evidence_status="unverified"` + `metadata.simulated=True` (nhu cli.py:237-239 da lam dung). |
| 2 | CAO | lineage/impact.py:137-163 | Canh MLflow tu run duoc gan cung `evidence_status="verified"` bat ke `run_origin`; dong 266-267 run khong ro origin bi xep vao `actual_mlflow`. | Dung `run.run_origin == "actual_mlflow"` moi verified; con lai unverified; origin la la `unknown`. |
| 3 | CAO | adapters/cvat_adapter.py:248, 341 | Manifest sinh ra hard-code `"privacy_cleared": True` cho moi anh (coco250 & cvat cache edufun) -> bia ket qua privacy, validator co the pass nho co nay. | De `None`/`False` (chua duyet); chi True khi co ApprovalRecord privacy that. |
| 4 | CAO | dashboard/app.py:571-646 | Endpoint approve/publish/accept khong xac thuc, khong guard RC_READ_ONLY, khong ghi audit; `actor`/`consumer` la chuoi tu do -> ai goi API cung "ky" qc+privacy+ACK; approve tu nhay APPROVED (dong 582-590). | Them AuditLog + yeu cau header/nguoi duyet that; khoi auto-approve; bind 127.0.0.1 (da mac dinh o cli.py:115) va chan ghi khi RC_READ_ONLY. |
| 5 | CAO | cli.py:260-263, 307-308 | Mot `--approver` duy nhat ghi dong thoi qc va privacy APPROVED (note "CLI --approver") -> mot nguoi tu duyet hai vai tro. | Yeu cau `--qc-approver` va `--privacy-approver` rieng, khac nhau (hoac xac nhan co chu ky). |
| 6 | TB | config.py:41-43 + toan repo | `guard_cvat_write` chi duoc test goi, khong module nao goi; `RC_READ_ONLY` khong chan gi thuc te. Khong co write-guard host/task trong code cu (da co `audit.guard_write` moi nhung chua duoc noi). | Goi `audit.guard_write` truoc moi thao tac ghi CVAT (CvatDemoClient.put_annotations); ghi audit moi lenh CLI. |
| 7 | TB | cli.py:316, app.py:625-646 | `accept` khong bat buoc `verified_sha256` (None) -> ACK khong gan voi digest bao ve. | Bat buoc consumer cung cap digest va so sanh voi snapshot cua release. |
| 8 | TB | cli.py:195, versions/restore.py:59, fault_lab/inject.py:110, demo/run_e2e.py:122 | `shutil.rmtree` tren duong dan nguoi dung truyen (`restore <target_dir>`) khong kiem tra nam trong artifacts/ -> co the xoa thu muc tuy y. | Chi rmtree khi target nam duoi `config.artifacts_dir`; nguoc lai tu choi. |
| 9 | TB | dashboard/app.py:155-156 | `_normalize_exported_manifest` tu sinh `classes = class_0..79` khi thieu -> bia taxonomy. | Doc taxonomy that (`contracts/taxonomy.yaml`) hoac bao thieu. |
| 10 | TB | adapters/cvat_adapter.py:90-95 | `pat="configured"` truyen vao CvatGateway la gia tri gia (khong lo secret, nhung de gay nham); `cvat_jobs_dir` mac dinh tro `jobs/edufun` (du lieu cvat.edufun.ai) dung de export du lieu that vao release. | Doi ten tham so/placeholder ro rang; cach ly du lieu edufun khoi demo (Agent Console chi dung local). |
| 11 | THAP | lineage/events.py:99-104 | `evidence_status` chi la co trong metadata, khong yeu cau `evidence` (duong dan/digest) khi `verified`. | Bat buoc truong `evidence` khi verified. |
| 12 | THAP | config.py:46-49,adapters/cvat_adapter.py:15-22 | Duong dan tuyet doi `D:\06-Learning...` hard-code lam default. | Doc tu env/relative, bo default tuyet doi. |

## Khong tim thay
- Khong thay code ghi len `cvat.edufun.ai` (adapter dat `write_enabled=False`; CLI khong goi API ghi CVAT). Rui ro chi o #6 (chua co guard that khi CvatDemoClient duoc them).
- Khong thay approval/ACK/training bia tu dong trong luong CLI mac dinh (chi khi co `--approver/--consumer`); rui ro bia nam o #1, #2, #3, #4.
