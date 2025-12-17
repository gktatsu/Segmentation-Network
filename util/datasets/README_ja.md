# データセット分割・整形ツール

## 概要

このリポジトリには、ペア画像（例: セグメンテーションの入力画像と対応マスク）を学習・検証・テストに振り分けるツール群が含まれています。すべて標準ライブラリのみで動作し、Linux / macOS / Windows いずれの環境でも Python 3.7 以上で利用できます。

ツールは `util/datasets/random_split/` 配下に集約されています。親ディレクトリ側には互換目的のラッパーのみ残しています。

---

## ファイル一覧

| ファイル | 役割 |
|----------|------|
| `random_split/split_dataset.py` | 複数の画像ディレクトリとマスクディレクトリを照合し、指定比率で `train` / `valid` / `test` に分割。ログ（CSV / summary）を生成。 |
| `random_split/format_dataset.py` | `split_dataset.py` の出力を最終提出形式（`train_images/` など）にコピー＆連番リネーム。 |
| `random_split/run_split_and_format.sh` | 分割と整形を連続実行するシェルラッパー。分割完了後に中間ディレクトリを削除し、整形済みフォルダのみを残す。 |
| `random_split/rename_images.py` | 任意ディレクトリ内の画像を連番リネーム。`--recursive` で配下を一括処理可能。 |
| `random_split/README.md` | 各スクリプトのクイックリファレンス。 |

---

## 必要条件

- Python 3.7 以上
- 画像とマスクで拡張子を除いたベース名が一致していること（例: `image1.png` と `image1.png`）
- 入力ディレクトリは非再帰でスキャン。サブフォルダを含めたい場合は事前にフラット化するか `rename_images.py --recursive` を利用

---

## 典型的なワークフロー

### 1. 分割の確認 (dry-run)

```bash
python3 random_split/split_dataset.py \
    --images /path/to/images \
    --masks  /path/to/masks \
    --out    /path/to/out_dir \
    --train 0.6 --valid 0.2 --test 0.2 \
    --dry-run
```

- コピーや移動は行われず、ログのみ生成されます。

### 2. 本実行

```bash
python3 random_split/split_dataset.py \
    --images /path/to/images \
    --masks  /path/to/masks \
    --out    /path/to/out_dir \
    --train 0.6 --valid 0.2 --test 0.2
```

- デフォルトではコピー。`--move` を付けると移動。
- 出力ディレクトリ名には自動でタイムスタンプ（`_yyyymmddHHMM`）が付与されます。`--no-append-timestamp` で無効化可能。
- 実行完了時に `SPLIT_DATASET_OUTPUT_ROOT=/actual/path` が出力されるため、後続処理で利用できます。

### 3. 既存のテストセットを保持して再分配

```bash
python3 random_split/split_dataset.py \
    --images /path/to/train_images /path/to/test_images \
    --masks  /path/to/train_masks  /path/to/test_masks \
    --out    /path/to/out_dir \
    --train 0.6 --valid 0.2 --test 0.2 \
    --preserve-original-test
```

- 2 番目の入力（index 1）のペアはそのまま `test` に残り、残りを `train` / `valid` で再分配します。
- `--preserve-test-index N` で任意のインデックスを指定することも可能。

### 4. 最終フォーマットへの整形

```bash
python3 random_split/format_dataset.py \
    --source /path/to/out_dir_202512171234 \
    --dest   /path/to/final_dataset
```

- `final_dataset/` 直下に `train_images/`, `train_masks/`, `validation_images/`, `validation_masks/`, `test_images/`, `test_masks/` が生成されます。
- 各サブセットで `image1.png`, `mask1.png` のような連番にリネームされます。

### 5. 分割と整形をまとめて実行

```bash
bash random_split/run_split_and_format.sh \
    --images /path/to/images \
    --masks  /path/to/masks \
    --out    /path/to/out_dir \
    --train 0.6 --valid 0.2 --test 0.2
```

- `--format-dest /your/dest` で整形出力先を指定可能。省略時は `<split出力>_formatted`。
- `PYTHON_BIN` 環境変数で Python インタプリタを切り替え可能。
- `--dry-run` を付けると分割のみをドライランで実行し、整形はスキップ。
- 正常完了時は中間の split ディレクトリを削除し、ログを整形済みフォルダに移動。

---

## フォルダ構造イメージ

```
split_out_YYYYMMDDHHMM/         # split_dataset.py の出力
├── train/
│   ├── images/
│   └── masks/
├── valid/
│   ├── images/
│   └── masks/
├── test/
│   ├── images/
│   └── masks/
├── split_log_*.csv
└── split_summary_*.txt

Dataset/                        # format_dataset.py の出力
├── train_images/
├── train_masks/
├── validation_images/
├── validation_masks/
├── test_images/
└── test_masks/
```

---

## `split_dataset.py` オプション一覧

| オプション | 役割 |
|-------------|------|
| `--images` (必須) | 画像ディレクトリ（複数指定可）。同じベース名でマージ。 |
| `--masks` (必須) | マスクディレクトリ（複数指定可）。 |
| `--out` (必須) | 出力ルート。`train/valid/test` 配下に `images/`・`masks/` を作成。 |
| `--train`, `--valid`, `--test` | 分割比。少数でも百分率でも可。合計が自動正規化。 |
| `--seed` | 乱数シード（デフォルト 42）。 |
| `--move` | コピーではなく移動。 |
| `--dry-run` | ファイル操作を行わずログのみ生成。 |
| `--preserve-test-index` | 指定インデックスの入力セットを `test` として保持。 |
| `--preserve-original-test` | 入力が 2 組のときに index 1 を自動で `test` 保存。 |
| `--no-append-timestamp` | 出力ルートへのタイムスタンプ付与を無効化。 |

---

## `format_dataset.py` オプション一覧

| オプション | 役割 |
|-------------|------|
| `--source` (必須) | `split_dataset.py` の出力ルート。 |
| `--dest` (必須) | 整形後の出力先。存在しない場合は自動作成。 |
| `--dry-run` | コピーを行わず、実行内容のみ表示。 |

---

## `rename_images.py` — 画像連番リネームツール

単一ディレクトリまたは再帰的に複数ディレクトリ内の画像を連番リネームします。

### 基本コマンド

```bash
python3 random_split/rename_images.py \
    --dir /path/to/images \
    --prefix sample \
    --start 1 \
    --zero-pad 4
```

| オプション | 説明 |
|-----------|------|
| `--dir` | 対象ディレクトリ（必須）。 |
| `--prefix` | ファイル名の接頭辞（デフォルト `gen_img`）。 |
| `--start` | 連番の開始番号（デフォルト 0）。 |
| `--zero-pad` | ゼロ埋め幅。`--auto-pad` で最大番号に応じて自動調整。 |
| `--ext` | 対象拡張子を複数指定可（デフォルト `.png`）。 |
| `--output` | 別ディレクトリにコピー。省略時はインプレースリネーム。 |
| `--recursive` | `--dir` 配下のサブディレクトリを一括処理。 |
| `--skip-processed` | マーカーファイルがあるディレクトリはスキップ。 |
| `--write-marker` | 正常終了後にマーカーを作成。 |
| `--marker-name` | マーカーファイル名（デフォルト `.rename_images_done`）。 |
| `--overwrite` | 既存ファイルの上書きを許可。 |
| `--dry-run` | 変更せず計画のみ表示。 |

### 再帰モード

```bash
python3 random_split/rename_images.py \
    --dir /datasets/raw_assets \
    --recursive \
    --prefix dataset \
    --start 0 \
    --auto-pad
```

- 「指定拡張子のファイルのみを含むディレクトリ」を自動検出して一括処理。
- `--output` と組み合わせるとルートからの相対パス構造を保ったまま別ツリーに複製。
- 連番は各ディレクトリで独立してリセット。

---

## 出力されるログ

`split_dataset.py` 実行時に `--out` 直下に生成：

- `split_log_<timestamp>.csv` — 各ファイルの割当先・元パスを記録
- `split_summary_<timestamp>.txt` — 総ペア数、train/valid/test 件数

`format_dataset.py` はログを生成しませんが、各サブセットのコピー件数を標準出力に表示します。

---

## 実装上の注意

- 重複ベース名が複数の入力ディレクトリで見つかった場合はエラーで停止。
- 余剰サンプルは `train` に割り当て。別ルールが必要な場合は `compute_counts` を変更。
- `format_dataset.py` はベース名が一致しないペアを自動スキップし、共通部分のみ整形。

---

## トラブルシューティング

| 症状 | 原因・対処 |
|------|------------|
| `No paired files found` | ベース名が一致していない、または片方しか存在しない。命名規則と入力パスを確認。 |
| `Duplicate basenames detected` | 複数入力ディレクトリに同名ペアが存在。ファイルの整理が必要。 |
| ログはあるがファイルがコピーされない | `--dry-run` を付けたまま実行している可能性。 |
| 目的の分割比にならない | サンプル数が少ないと丸めで偏りが出る。サンプル数を増やすか `compute_counts` を調整。 |

---

## 今後の拡張アイデア

- 再帰的なサブフォルダ探索（`split_dataset.py` 向け）
- 余剰サンプルのランダム再配分
- コピー完了後のハッシュ整合性チェック
- 進捗バー（`tqdm`）や並列コピーの導入

---

ご自身の環境に合わせた具体的なコマンド例や、追加機能の要望があれば遠慮なく共有してください。

