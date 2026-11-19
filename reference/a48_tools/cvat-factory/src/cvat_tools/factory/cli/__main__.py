"""Module entrypoint for `python -m cvat_tools.factory.cli`."""

from __future__ import annotations

import sys

from cvat_tools.factory.cli.app import main

if __name__ == "__main__":
    sys.exit(main())
