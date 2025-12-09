#!/usr/bin/env python3
"""Batch runner for copy_batch_files.py.

This script reads a configuration file (CSV or JSON) and executes
copy_batch_files.py for each source/destination pair.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run copy_batch_files.py for multiple source/destination pairs "
            "defined in a configuration file (CSV or JSON)."
        )
    )
    parser.add_argument(
        "config_file",
        type=Path,
        help="Path to configuration file (CSV or JSON).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to each copy_batch_files.py invocation.",
    )
    parser.add_argument(
        "--max-index",
        type=int,
        default=None,
        help=(
            "Override max_index for all entries. "
            "If not specified, uses per-entry max_index from config."
        ),
    )
    return parser.parse_args()


def load_csv_config(config_path: Path) -> List[Dict[str, Any]]:
    """Load configuration from a CSV file.

    Expected columns: source_dir, destination_dir, max_index (optional)
    """
    entries: List[Dict[str, Any]] = []
    with open(config_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry: Dict[str, Any] = {
                "source_dir": row["source_dir"].strip(),
                "destination_dir": row["destination_dir"].strip(),
            }
            if "max_index" in row and row["max_index"].strip():
                entry["max_index"] = int(row["max_index"].strip())
            entries.append(entry)
    return entries


def load_json_config(config_path: Path) -> List[Dict[str, Any]]:
    """Load configuration from a JSON file.

    Expected format:
    [
        {
            "source_dir": "/path/to/source",
            "destination_dir": "/path/to/dest",
            "max_index": 1
        },
        ...
    ]
    """
    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON config must be a list of entries")
    return data


def load_config(config_path: Path) -> List[Dict[str, Any]]:
    """Load configuration from CSV or JSON based on file extension."""
    suffix = config_path.suffix.lower()
    if suffix == ".csv":
        return load_csv_config(config_path)
    elif suffix == ".json":
        return load_json_config(config_path)
    else:
        raise ValueError(
            f"Unsupported config file format: {suffix}. Use .csv or .json"
        )


def run_copy_batch_files(
    source_dir: str,
    destination_dir: str,
    max_index: int,
    dry_run: bool,
) -> int:
    """Run copy_batch_files.py as a subprocess."""
    script_path = Path(__file__).parent / "copy_batch_files.py"
    cmd = [
        sys.executable,
        str(script_path),
        source_dir,
        destination_dir,
        "--max_index",
        str(max_index),
    ]
    if dry_run:
        cmd.append("--dry-run")

    print(f"\n{'=' * 60}")
    print(f"Source: {source_dir}")
    print(f"Destination: {destination_dir}")
    print(f"Max Index: {max_index}")
    print(f"{'=' * 60}")

    result = subprocess.run(cmd, check=False)
    return result.returncode


def main() -> None:
    args = parse_args()
    config_path: Path = args.config_file.expanduser().resolve()

    if not config_path.exists():
        raise SystemExit(f"Config file does not exist: {config_path}")

    entries = load_config(config_path)
    if not entries:
        print("No entries found in configuration file.")
        return

    print(f"Loaded {len(entries)} entries from {config_path}")

    success_count = 0
    fail_count = 0

    for i, entry in enumerate(entries, 1):
        source_dir = entry.get("source_dir")
        destination_dir = entry.get("destination_dir")

        if not source_dir or not destination_dir:
            print(f"[{i}] Skipping invalid entry: {entry}")
            fail_count += 1
            continue

        # Use command-line override if provided, else use per-entry value
        if args.max_index is not None:
            max_index = args.max_index
        elif "max_index" in entry:
            max_index = entry["max_index"]
        else:
            print(f"[{i}] Skipping entry without max_index: {entry}")
            fail_count += 1
            continue

        print(f"\n[{i}/{len(entries)}] Processing...")
        returncode = run_copy_batch_files(
            source_dir=source_dir,
            destination_dir=destination_dir,
            max_index=max_index,
            dry_run=args.dry_run,
        )

        if returncode == 0:
            success_count += 1
        else:
            fail_count += 1
            print(f"[{i}] Failed with exit code {returncode}")

    print("\n" + "=" * 60)
    print("Batch processing complete.")
    print(f"Success: {success_count}, Failed: {fail_count}")
    print(f"{'=' * 60}")

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
