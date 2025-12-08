#!/usr/bin/env python3
"""Copy batch files from a source directory to a destination directory.

This utility copies every file whose *filename* (case-insensitive) contains
``batch{i}`` for any integer ``i`` between 0 and ``max_index`` (inclusive).
Directory structure relative to the source root is preserved.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy files whose names contain batch[i] (case-insensitive) for "
            "0 <= i <= max_index. Directory structure is preserved."
        )
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Directory to search for files to copy.",
    )
    parser.add_argument(
        "destination_dir",
        type=Path,
        help="Directory where matching files will be copied to.",
    )
    parser.add_argument(
        "max_index",
        type=int,
        help="Highest batch index to include (copies batch0 through batchN).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List which files *would* be copied without copying them.",
    )
    return parser.parse_args()


def validate_args(source: Path, destination: Path, max_index: int) -> None:
    if not source.exists():
        raise SystemExit(f"Source directory does not exist: {source}")
    if not source.is_dir():
        raise SystemExit(f"Source path is not a directory: {source}")
    if max_index < 0:
        raise SystemExit("max_index must be a non-negative integer")
    try:
        destination.resolve().relative_to(source.resolve())
    except ValueError:
        pass
    else:
        raise SystemExit("Destination cannot be inside the source directory")


def collect_candidates(source: Path, max_index: int) -> List[Path]:
    tokens = tuple(f"batch{i}" for i in range(max_index + 1))
    matches: List[Path] = []
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        name_lower = path.name.lower()
        if any(token in name_lower for token in tokens):
            matches.append(path)
    return matches


def copy_files(
    files: Iterable[Path],
    source: Path,
    destination: Path,
    dry_run: bool,
) -> int:
    copied = 0
    for file_path in files:
        relative = file_path.relative_to(source)
        target = destination / relative
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, target)
        prefix = "[DRY-RUN] " if dry_run else ""
        print(f"{prefix}Copying {file_path} -> {target}")
        copied += 1
    return copied


def main() -> None:
    args = parse_args()
    source: Path = args.source_dir.expanduser().resolve()
    destination: Path = args.destination_dir.expanduser().resolve()
    max_index: int = args.max_index

    validate_args(source, destination, max_index)

    matches = collect_candidates(source, max_index)
    if not matches:
        print("No files matched the batch pattern.")
        return

    count = copy_files(matches, source, destination, args.dry_run)
    suffix = " (dry run)" if args.dry_run else ""
    print(f"Copied {count} file(s){suffix}.")


if __name__ == "__main__":
    main()
