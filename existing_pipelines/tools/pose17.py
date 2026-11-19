"""Thin shim for `python -m tools.pose17 ...` delegating to `cvat_tools.pose.pose17_cli`."""
from __future__ import annotations

import sys

from cvat_tools.pose.pose17_cli import main

if __name__ == "__main__":
    sys.exit(main())
