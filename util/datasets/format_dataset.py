#!/usr/bin/env python3
"""分割済みデータセットを所定のフォルダ構造とファイル名に整形するツール。"""
import argparse
import os
import shutil
from typing import Dict, List, Tuple

SubsetMap = Dict[str, Tuple[str, str]]


def list_files(dirpath: str) -> List[str]:
    """指定ディレクトリ直下のファイル一覧を取得する。

    Args:
        dirpath (str): 走査対象のディレクトリパス。

    Returns:
        list[str]: ディレクトリ直下に存在するファイル名のリスト。
    """
    try:
        return [
            f
            for f in os.listdir(dirpath)
            if os.path.isfile(os.path.join(dirpath, f))
        ]
    except FileNotFoundError:
        return []


def basename_no_ext(filename: str) -> str:
    """拡張子を除いたベース名を取得する。

    Args:
        filename (str): ファイル名またはパス。

    Returns:
        str: 拡張子を除いたベース名。
    """
    return os.path.splitext(filename)[0]


def ensure_dir(path: str) -> None:
    """ディレクトリが存在しない場合は作成する。

    Args:
        path (str): 作成するディレクトリパス。
    """
    os.makedirs(path, exist_ok=True)


def pair_files(images: List[str], masks: List[str]) -> List[Tuple[str, str]]:
    """画像とマスクのファイル名からペアを作成する。

    Args:
        images (list[str]): 画像ファイル名のリスト。
        masks (list[str]): マスクファイル名のリスト。

    Returns:
        list[tuple[str, str]]: ベース名が一致した画像・マスクの組み合わせ。
    """
    img_map = {basename_no_ext(f): f for f in images}
    mask_map = {basename_no_ext(f): f for f in masks}
    common = sorted(img_map.keys() & mask_map.keys())
    pairs: List[Tuple[str, str]] = []
    for key in common:
        pairs.append((img_map[key], mask_map[key]))
    return pairs


def copy_pair(
    src_img_dir: str,
    src_mask_dir: str,
    dest_img_dir: str,
    dest_mask_dir: str,
    img_name: str,
    mask_name: str,
    new_img_name: str,
    new_mask_name: str,
    dry_run: bool,
) -> None:
    """画像とマスクのペアをコピーし、必要に応じてリネームする。

    Args:
        src_img_dir (str): 画像の元ディレクトリ。
        src_mask_dir (str): マスクの元ディレクトリ。
        dest_img_dir (str): 画像のコピー先ディレクトリ。
        dest_mask_dir (str): マスクのコピー先ディレクトリ。
        img_name (str): 元の画像ファイル名。
        mask_name (str): 元のマスクファイル名。
        new_img_name (str): 出力する画像ファイル名。
        new_mask_name (str): 出力するマスクファイル名。
        dry_run (bool): True の場合はコピーを行わない。
    """
    ensure_dir(dest_img_dir)
    ensure_dir(dest_mask_dir)
    src_img = os.path.join(src_img_dir, img_name)
    src_mask = os.path.join(src_mask_dir, mask_name)
    dest_img = os.path.join(dest_img_dir, new_img_name)
    dest_mask = os.path.join(dest_mask_dir, new_mask_name)
    if dry_run:
        return
    shutil.copy2(src_img, dest_img)
    shutil.copy2(src_mask, dest_mask)


def format_dataset(source_root: str, dest_root: str, dry_run: bool) -> None:
    """分割済みデータセットを指定フォルダ構成へ整形する。

    Args:
        source_root (str): `train/valid/test` を含む入力ルート。
        dest_root (str): 整形後の出力ルート。
        dry_run (bool): True の場合はコピーを行わず内容のみ表示する。
    """
    subset_config: SubsetMap = {
        "train": ("train_images", "train_masks"),
        "valid": ("validation_images", "validation_masks"),
        "test": ("test_images", "test_masks"),
    }

    for subset, (img_dir_name, mask_dir_name) in subset_config.items():
        src_img_dir = os.path.join(source_root, subset, "images")
        src_mask_dir = os.path.join(source_root, subset, "masks")
        dest_img_dir = os.path.join(dest_root, img_dir_name)
        dest_mask_dir = os.path.join(dest_root, mask_dir_name)

        images = list_files(src_img_dir)
        masks = list_files(src_mask_dir)
        pairs = pair_files(images, masks)

        if not pairs:
            print(f"No paired files found for subset '{subset}'")
            continue

        for idx, (img_name, mask_name) in enumerate(pairs, start=1):
            img_ext = os.path.splitext(img_name)[1]
            mask_ext = os.path.splitext(mask_name)[1]
            new_img = f"image{idx}{img_ext}"
            new_mask = f"mask{idx}{mask_ext}"
            copy_pair(src_img_dir, src_mask_dir, dest_img_dir, dest_mask_dir,
                      img_name, mask_name, new_img, new_mask, dry_run)

        print(f"Formatted {len(pairs)} pairs for subset '{subset}'")

    if dry_run:
        print("Dry-run complete (no files were copied).")


def parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。

    Returns:
        argparse.Namespace: 解析結果の名前空間。
    """
    parser = argparse.ArgumentParser(
        description="Reformat dataset folder and filenames"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Root directory of split dataset (train/valid/test subfolders)",
    )
    parser.add_argument(
        "--dest",
        required=True,
        help="Destination root directory for formatted dataset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show actions without copying files",
    )
    return parser.parse_args()


def main() -> None:
    """エントリーポイント。引数を解析して整形処理を呼び出す。"""
    args = parse_args()
    format_dataset(args.source, args.dest, args.dry_run)


if __name__ == "__main__":
    main()
