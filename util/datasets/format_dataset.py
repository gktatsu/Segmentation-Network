#!/usr/bin/env python3
"""Compatibility shim forwarding to ``random_split/format_dataset.py``.

Existing calls to ``util/datasets/format_dataset.py`` keep working, while
the real implementation lives under ``random_split``.
"""
from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("random_split") / "format_dataset.py"
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

