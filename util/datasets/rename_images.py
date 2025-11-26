#!/usr/bin/env python3
"""指定ディレクトリ内の画像ファイルを連番リネームするツール。"""
from __future__ import annotations

import argparse
import os
import sys
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple
from uuid import uuid4


def normalize_extensions(exts: Sequence[str] | None) -> List[str] | None:
    """拡張子の表記ゆれを揃える。"""
    if not exts:
        return None
    normalized: List[str] = []
    for ext in exts:
        ext = ext.strip()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.append(ext.lower())
    return normalized or None


def iter_target_files(
    directory: Path, extensions: List[str] | None
) -> List[Path]:
    """対象ディレクトリからリネーム対象ファイルを取得する。"""
    files: List[Path] = []
    for entry in sorted(directory.iterdir()):
        if not entry.is_file():
            continue
        if extensions and entry.suffix.lower() not in extensions:
            continue
        files.append(entry)
    return files


def build_operations(
    files: Sequence[Path],
    prefix: str,
    start_index: int,
    zero_pad: int,
    auto_pad: bool,
    output_dir: Path | None,
) -> List[Tuple[Path, Path]]:
    """(元 -> 先) のリネーム計画を生成する。"""
    total = len(files)
    if total == 0:
        return []
    max_index = start_index + total - 1
    pad_width = max(zero_pad, 0)
    if auto_pad:
        pad_width = max(pad_width, len(str(max_index)))
    operations: List[Tuple[Path, Path]] = []
    for offset, src in enumerate(files):
        idx = start_index + offset
        number = f"{idx:0{pad_width}d}" if pad_width > 0 else str(idx)
        new_name = f"{prefix}_{number}{src.suffix}"
        target_dir = output_dir if output_dir is not None else src.parent
        operations.append((src, target_dir / new_name))
    return operations


def ensure_no_conflicts(
    operations: Sequence[Tuple[Path, Path]],
    overwrite: bool,
    copy_mode: bool,
) -> None:
    """リネーム前に競合がないか確認する。"""
    if copy_mode:
        for _, dest in operations:
            if dest.exists() and not overwrite:
                raise FileExistsError(
                    (
                        f"Destination '{dest}' already exists. "
                        "Use --overwrite to replace it."
                    )
                )
        return
    source_paths = {src for src, _ in operations}
    for src, dest in operations:
        if dest == src:
            continue
        if dest.exists() and dest not in source_paths and not overwrite:
            raise FileExistsError(
                (
                    f"Destination '{dest}' already exists. "
                    "Use --overwrite to replace it."
                )
            )


def apply_operations(
    operations: Sequence[Tuple[Path, Path]],
    dry_run: bool,
    overwrite: bool,
    copy_mode: bool,
) -> None:
    """二段階リネームで安全に適用する。"""
    if dry_run:
        for src, dest in operations:
            if src == dest:
                continue
            print(f"DRY-RUN: {src.name} -> {dest.name}")
        print("Dry-run complete (no files were modified).")
        return

    if copy_mode:
        for src, dest in operations:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if overwrite:
                    if dest.is_file():
                        dest.unlink()
                    else:
                        raise IsADirectoryError(
                            f"Destination '{dest}' exists and is not a file."
                        )
                else:
                    raise FileExistsError(
                        (
                            f"Destination '{dest}' already exists. "
                            "Use --overwrite to replace it."
                        )
                    )
            shutil.copy2(src, dest)
        return

    temp_moves: List[Tuple[Path, Path]] = []
    for idx, (src, dest) in enumerate(operations):
        if src == dest:
            continue
        temp_name = src.with_name(
            f".__tmp_{uuid4().hex}_{idx}{src.suffix}"
        )
        while temp_name.exists():
            temp_name = src.with_name(
                f".__tmp_{uuid4().hex}_{idx}{src.suffix}"
            )
        src.rename(temp_name)
        temp_moves.append((temp_name, dest))

    for temp, dest in temp_moves:
        if dest.exists():
            if overwrite:
                if dest.is_file():
                    dest.unlink()
                else:
                    raise IsADirectoryError(
                        f"Destination '{dest}' exists and is not a file."
                    )
            else:
                raise FileExistsError(
                    (
                        f"Destination '{dest}' unexpectedly exists. "
                        "Use --overwrite to replace it."
                    )
                )
        temp.rename(dest)


def is_image_only_directory(
    directory: Path, extensions: List[str] | None
) -> bool:
    """ディレクトリが指定拡張子のファイルのみを含むかを判定する。"""

    try:
        files: List[Path] = []
        has_subdirs = False
        for entry in directory.iterdir():
            if entry.is_file():
                files.append(entry)
            elif entry.is_dir():
                has_subdirs = True
            # シンボリックリンクなどは無視
        if has_subdirs:
            return False
        if not files:
            return False
        if not extensions:
            return True
        return all(file.suffix.lower() in extensions for file in files)
    except PermissionError:
        return False


def collect_image_only_dirs(
    root: Path, extensions: List[str] | None
) -> List[Path]:
    """ルート配下で画像のみを含むディレクトリ一覧を取得する。"""

    targets: List[Path] = []
    for dirpath, _dirnames, _filenames in os.walk(root):
        directory = Path(dirpath)
        if is_image_only_directory(directory, extensions):
            targets.append(directory)
    return targets


def resolve_output_subdir(
    base_output: Path | None, root: Path, current: Path
) -> Path | None:
    """再帰処理時にルートからの相対パスに応じた出力先を決定する。"""

    if base_output is None:
        return None
    try:
        relative = current.relative_to(root)
    except ValueError:
        # 念のため root 外のディレクトリが来てもそのまま返す
        return base_output
    if relative == Path('.'):
        return base_output
    return base_output / relative


def rename_images(
    directory: Path,
    prefix: str,
    start_index: int,
    zero_pad: int,
    auto_pad: bool,
    extensions: List[str] | None,
    dry_run: bool,
    overwrite: bool,
    output_dir: Path | None,
) -> int:
    """メイン処理。"""
    files = iter_target_files(directory, extensions)
    if not files:
        print("No matching files were found.")
        return 0
    copy_mode = output_dir is not None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    operations = build_operations(
        files, prefix, start_index, zero_pad, auto_pad, output_dir
    )
    ensure_no_conflicts(operations, overwrite, copy_mode)
    apply_operations(operations, dry_run, overwrite, copy_mode)
    renamed = sum(1 for src, dest in operations if src != dest)
    print(f"Processed {len(files)} files. Renamed {renamed} of them.")
    return renamed


def rename_images_recursive(
    root_dir: Path,
    prefix: str,
    start_index: int,
    zero_pad: int,
    auto_pad: bool,
    extensions: List[str] | None,
    dry_run: bool,
    overwrite: bool,
    output_dir: Path | None,
) -> int:
    """ルート配下の全サブディレクトリに対して連番リネームを適用する。"""

    targets = collect_image_only_dirs(root_dir, extensions)
    if not targets:
        print(
            "No subdirectories containing only the target image extensions were found."  # noqa: E501
        )
        return 0

    total_dirs = 0
    total_renamed = 0
    for directory in sorted(targets):
        dest_dir = resolve_output_subdir(output_dir, root_dir, directory)
        print(f"[recursive] Processing {directory}")
        renamed = rename_images(
            directory=directory,
            prefix=prefix,
            start_index=start_index,
            zero_pad=zero_pad,
            auto_pad=auto_pad,
            extensions=extensions,
            dry_run=dry_run,
            overwrite=overwrite,
            output_dir=dest_dir,
        )
        total_dirs += 1
        total_renamed += renamed

    print(
        f"Recursive rename completed: processed {total_dirs} directories, "
        f"renamed {total_renamed} files in total."
    )
    return total_renamed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rename images in a directory to sequential names "
            "(e.g. img_0001.png)"
        ),
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Target directory containing images to rename",
    )
    parser.add_argument(
        "--prefix",
        default="gen_img",
        help="Prefix for renamed files (default: gen_img)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Starting index for numbering (default: 0)",
    )
    parser.add_argument(
        "--zero-pad",
        type=int,
        default=0,
        help="Fixed zero padding width (default: 0)",
    )
    parser.add_argument(
        "--auto-pad",
        action="store_true",
        help="Automatically pad based on the max index",
    )
    parser.add_argument(
        "--ext",
        action="append",
        default=[".png"],
        help="Extension(s) to include (default: .png). Repeatable.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Directory where renamed files are written. "
            "Defaults to in-place renaming."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Process every subdirectory under --dir that contains only files "
            "with the target extensions."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned renames without modifying files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting files that already use the target name",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    target_dir = Path(args.dir).expanduser().resolve()
    if not target_dir.is_dir():
        print(f"Error: '{target_dir}' is not a directory.")
        return 1
    extensions = normalize_extensions(args.ext)
    output_dir = (
        Path(args.output).expanduser().resolve() if args.output else None
    )
    if args.recursive:
        rename_images_recursive(
            root_dir=target_dir,
            prefix=args.prefix,
            start_index=args.start,
            zero_pad=args.zero_pad,
            auto_pad=args.auto_pad,
            extensions=extensions,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            output_dir=output_dir,
        )
    else:
        rename_images(
            directory=target_dir,
            prefix=args.prefix,
            start_index=args.start,
            zero_pad=args.zero_pad,
            auto_pad=args.auto_pad,
            extensions=extensions,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            output_dir=output_dir,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
