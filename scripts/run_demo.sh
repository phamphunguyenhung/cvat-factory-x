#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
else
  echo "Không có .env (dùng giá trị mặc định / auto-discover)."
fi

python -X utf8 -m release_control.cli demo

echo
echo "Mở dashboard:"
echo "  python -X utf8 -m uvicorn dashboard.app:app --port 8765"
