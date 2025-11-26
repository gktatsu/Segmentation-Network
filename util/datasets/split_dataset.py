#!/usr/bin/env python3
"""画像とマスクのペアを学習・検証・テスト用に分割する CLI ツール。"""
import argparse
import csv
import os
import random
import shutil
import sys
from datetime import datetime


def list_files_nonrec(dirpath):
    """指定ディレクトリ直下のファイル一覧を取得する。

    Args:
        dirpath (str): 走査するディレクトリのパス。

    Returns:
        list[str]: ディレクトリ直下に存在するファイル名のリスト。
    """
    try:
        return [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]
    except FileNotFoundError:
        return []


def basename_no_ext(filename):
    """拡張子を除いたベース名を取得する。

    Args:
        filename (str): ファイル名またはパス。

    Returns:
        str: 拡張子を除いたベース名。
    """
    return os.path.splitext(filename)[0]


def normalize_ratios(train, valid, test):
    """分割比率を正規化する。

    Args:
        train (float): 学習データの比率または百分率。
        valid (float): 検証データの比率または百分率。
        test (float): テストデータの比率または百分率。

    Returns:
        tuple[float, float, float]: 合計 1.0 に正規化された比率。

    Raises:
        ValueError: 比率の合計が 0 になる場合。
    """
    vals = [train, valid, test]
    # Allow percentages like 60, 20, 20
    if any(v > 1 for v in vals):
        s = sum(vals)
        if s == 0:
            raise ValueError("Ratios sum to zero")
        vals = [v / s for v in vals]
    else:
        s = sum(vals)
        if not (0.999 <= s <= 1.001):
            # normalize
            if s == 0:
                raise ValueError("Ratios sum to zero")
            vals = [v / s for v in vals]
    return tuple(vals)


def make_output_dirs(out_root):
    """出力先に train/valid/test 配下のディレクトリを作成する。

    Args:
        out_root (str): 出力ルートディレクトリのパス。
    """
    subsets = ["train", "valid", "test"]
    for s in subsets:
        images_dir = os.path.join(out_root, s, "images")
        masks_dir = os.path.join(out_root, s, "masks")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)


def compute_counts(n, ratios):
    """データ数と比率から各分割の枚数を算出する。

    Args:
        n (int): 全ペア数。
        ratios (tuple[float, float, float]): 学習・検証・テストの比率。

    Returns:
        list[int]: train, valid, test の順に並んだ枚数リスト。
    """
    # ratios: tuple of floats summing to ~1
    counts = [int(r * n) for r in ratios]
    remainder = n - sum(counts)
    # assign remainder to train (index 0)
    counts[0] += remainder
    return counts


def copy_or_move(src, dst, move=False, dry_run=False):
    """ファイルをコピーまたは移動する。

    Args:
        src (str): 元ファイルのパス。
        dst (str): 出力先のパス。
        move (bool): True の場合は移動、それ以外はコピー。
        dry_run (bool): True の場合は実際のファイル操作を行わない。
    """
    if dry_run:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if move:
        shutil.move(src, dst)
    else:
        shutil.copy2(src, dst)


def main():
    """コマンドライン引数を解析し分割処理を実行する。"""
    parser = argparse.ArgumentParser(
        description="Split paired image/mask dataset into train/valid/test")
    parser.add_argument("--images", required=True, nargs='+',
                        help="Path(s) to images folder(s) (non-recursive). Multiple allowed and will be merged")
    parser.add_argument("--masks", required=True, nargs='+',
                        help="Path(s) to masks folder(s) (non-recursive). Multiple allowed and will be merged")
    parser.add_argument("--out", required=True,
                        help="Output root path (will create train/valid/test under this)")
    parser.add_argument("--train", type=float, default=0.6,
                        help="Train ratio (default 0.6)")
    parser.add_argument("--valid", type=float, default=0.2,
                        help="Valid ratio (default 0.2)")
    parser.add_argument("--test", type=float, default=0.2,
                        help="Test ratio (default 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    parser.add_argument("--move", action="store_true",
                        help="Move files instead of copying")
    parser.add_argument("--dry-run", action="store_true",
                        help="Do not actually copy/move files, only produce the log")
    parser.add_argument("--preserve-original-test", action="store_true",
                        help="If set and exactly two input sources are provided, treat the second source (index 1) as preserved test set.")
    parser.add_argument("--preserve-test-index", type=int, default=None,
                        help="If set, the images/masks pair at this index (0-based) will be preserved as the test set; remaining sources are merged and split into train/valid")
    # By default append a run-timestamp to the output root dir (to avoid clobbering previous outputs).
    # Provide a flag to disable this behaviour.
    parser.add_argument("--no-append-timestamp", action="store_false", dest="append_timestamp",
                        help="Do not append run timestamp to the output root directory (default: append)")
    parser.set_defaults(append_timestamp=True)

    args = parser.parse_args()

    images_dirs = args.images
    masks_dirs = args.masks
    out_root = args.out
    ratios = normalize_ratios(args.train, args.valid, args.test)
    seed = args.seed
    move_flag = args.move
    dry_run = args.dry_run
    append_ts = args.append_timestamp
    # If enabled, append a run timestamp (yyyymmddHHMM) to the output root directory name only.
    # Example: /path/to/out -> /path/to/out_20251027HHMM
    if append_ts:
        run_ts = datetime.now().strftime('%Y%m%d%H%M')
        # normalise path (remove trailing slash) then append _timestamp to the final component
        out_root = os.path.normpath(out_root) + '_' + run_ts

    # Merge files from multiple input directories
    img_map = {}  # basename -> list of (images_dir, filename)
    any_images = False
    for images_dir in images_dirs:
        files = list_files_nonrec(images_dir)
        if files:
            any_images = True
        for f in files:
            img_map.setdefault(basename_no_ext(f), []).append((images_dir, f))

    mask_map = {}  # basename -> list of (masks_dir, filename)
    any_masks = False
    for masks_dir in masks_dirs:
        files = list_files_nonrec(masks_dir)
        if files:
            any_masks = True
        for f in files:
            mask_map.setdefault(basename_no_ext(f), []).append((masks_dir, f))

    if not any_images:
        print(f"No image files found in any of: {images_dirs}")
    if not any_masks:
        print(f"No mask files found in any of: {masks_dirs}")

    # Build list of basenames that have both image and mask
    paired_basenames = [b for b in img_map.keys() if b in mask_map]
    paired_basenames.sort()

    if not paired_basenames:
        print("No paired files found across provided directories — exiting")
        sys.exit(1)

    # Detect duplicates across sources: require exactly one image and one mask per basename
    dup_images = [b for b, v in img_map.items() if len(v) > 1]
    dup_masks = [b for b, v in mask_map.items() if len(v) > 1]
    if dup_images or dup_masks:
        print("Duplicate basenames detected across input directories. By default this tool requires unique basenames across all provided image dirs and mask dirs.")
        if dup_images:
            print(f"Examples of duplicate images (basename): {dup_images[:5]}")
        if dup_masks:
            print(f"Examples of duplicate masks (basename): {dup_masks[:5]}")
        print(
            "Please ensure basenames are unique or adjust the script to handle duplicates.")
        sys.exit(1)

    # Create pairs selecting the single entry for each basename
    # list of tuples: (basename, image_filename, mask_filename, image_dir, mask_dir)
    pairs_all = []
    for b in paired_basenames:
        img_dir, img_fname = img_map[b][0]
        mask_dir, mask_fname = mask_map[b][0]
        pairs_all.append((b, img_fname, mask_fname, img_dir, mask_dir))

    # Optionally preserve one source as the test set (do not merge it)
    preserve_index = args.preserve_test_index
    if args.preserve_original_test:
        # auto-preserve when exactly two sources provided
        if len(images_dirs) == 2 and len(masks_dirs) == 2:
            preserve_index = 1
        else:
            print(
                "--preserve-original-test requires exactly two input sources (images and masks)")
            sys.exit(1)

    preserved_pairs = []
    if preserve_index is not None:
        # Validate index
        if preserve_index < 0 or preserve_index >= len(images_dirs) or preserve_index >= len(masks_dirs):
            print(
                f"preserve-test-index {preserve_index} out of range for provided sources")
            sys.exit(1)
        # Filter pairs_all to find those whose source dirs match the preserve index dirs
        preserve_img_dir = images_dirs[preserve_index]
        preserve_mask_dir = masks_dirs[preserve_index]
        remaining_pairs = []
        for p in pairs_all:
            _, _, _, img_dir, mask_dir = p
            if os.path.abspath(img_dir) == os.path.abspath(preserve_img_dir) and os.path.abspath(mask_dir) == os.path.abspath(preserve_mask_dir):
                preserved_pairs.append(p)
            else:
                remaining_pairs.append(p)
        pairs = remaining_pairs
    else:
        pairs = pairs_all

    # Optionally preserve one source pair as the test set, otherwise merge-all
    preserve_idx = args.preserve_test_index

    # list of tuples: (basename, img_fname, mask_fname, img_dir, mask_dir, split)
    final_items = []

    if preserve_idx is not None:
        # validate index
        if preserve_idx < 0 or preserve_idx >= len(images_dirs) or preserve_idx >= len(masks_dirs):
            print(
                f"--preserve-test-index {preserve_idx} is out of range for provided input directories")
            sys.exit(1)

        preserved_img_dir = images_dirs[preserve_idx]
        preserved_mask_dir = masks_dirs[preserve_idx]

        preserved = [p for p in pairs if p[3] ==
                     preserved_img_dir and p[4] == preserved_mask_dir]
        remaining = [p for p in pairs if not (
            p[3] == preserved_img_dir and p[4] == preserved_mask_dir)]

        # assign preserved to test
        for (basename, img_fname, mask_fname, img_dir, mask_dir) in preserved:
            final_items.append(
                (basename, img_fname, mask_fname, img_dir, mask_dir, 'test'))

        # split remaining into train/valid only (preserved test counts as test)
        n_rem = len(remaining)
        if n_rem > 0:
            random.seed(seed)
            random.shuffle(remaining)
            tv_sum = args.train + args.valid
            if tv_sum <= 0:
                print('train+valid ratios must be > 0 when preserving a test source')
                sys.exit(1)
            train_norm = args.train / tv_sum
            # compute counts for remaining
            train_count = int(train_norm * n_rem)
            valid_count = n_rem - train_count
            # build splits for remaining
            splits_rem = ['train'] * train_count + ['valid'] * valid_count
            for item, s in zip(remaining, splits_rem):
                basename, img_fname, mask_fname, img_dir, mask_dir = item
                final_items.append(
                    (basename, img_fname, mask_fname, img_dir, mask_dir, s))
    else:
        # Shuffle and split all pairs into train/valid/test
        random.seed(seed)
        random.shuffle(pairs)

        n = len(pairs)
        counts = compute_counts(n, ratios)
        # counts corresponds to train, valid, test
        split_names = ["train", "valid", "test"]
        splits = []
        for split_idx, c in enumerate(counts):
            splits.extend([split_names[split_idx]] * c)
        # safety: if rounding produced mismatch, trim or extend
        if len(splits) > n:
            splits = splits[:n]
        elif len(splits) < n:
            splits.extend(["train"] * (n - len(splits)))

        for (basename, img_fname, mask_fname, img_dir, mask_dir), split in zip(pairs, splits):
            final_items.append(
                (basename, img_fname, mask_fname, img_dir, mask_dir, split))

    # Prepare output dirs
    make_output_dirs(out_root)

    # Prepare logs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(out_root, f"split_log_{timestamp}.csv")
    summary_path = os.path.join(out_root, f"split_summary_{timestamp}.txt")

    # Write CSV and perform copy/move (or dry-run) using final_items constructed above
    with open(csv_path, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "basename",
            "image_src",
            "image_src_dir",
            "mask_src",
            "mask_src_dir",
            "split",
            "dest_image",
            "dest_mask",
        ])

        for (basename, img_fname, mask_fname, img_dir, mask_dir, split) in final_items:
            src_image = os.path.join(img_dir, img_fname)
            src_mask = os.path.join(mask_dir, mask_fname)
            dest_image = os.path.join(out_root, split, "images", img_fname)
            dest_mask = os.path.join(out_root, split, "masks", mask_fname)

            writer.writerow([basename, src_image, img_dir,
                            src_mask, mask_dir, split, dest_image, dest_mask])

            try:
                copy_or_move(src_image, dest_image,
                             move=move_flag, dry_run=dry_run)
                copy_or_move(src_mask, dest_mask,
                             move=move_flag, dry_run=dry_run)
            except Exception as e:
                print(f"Failed to copy/move pair {basename}: {e}")

    # Write a simple summary file
    counts = {"train": 0, "valid": 0, "test": 0}
    for _b, _i, _m, _id, _md, s in final_items:
        counts[s] = counts.get(s, 0) + 1

    total = sum(counts.values())
    os.makedirs(out_root, exist_ok=True)
    with open(summary_path, "w", encoding='utf-8') as sf:
        sf.write(f"Total paired items: {total}\n")
        sf.write(f"Train: {counts.get('train', 0)}\n")
        sf.write(f"Valid: {counts.get('valid', 0)}\n")
        sf.write(f"Test: {counts.get('test', 0)}\n")

    print("Split finished.")
    print(f"CSV log: {csv_path}")
    print(f"Summary: {summary_path}")
    # Expose the resolved output root so wrapper scripts can chain downstream
    # steps without re-implementing the timestamp logic here.
    print(f"SPLIT_DATASET_OUTPUT_ROOT={out_root}")
    if dry_run:
        print("Dry-run: no files were copied/moved.")


if __name__ == '__main__':
    main()
