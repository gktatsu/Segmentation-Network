# random_split ユーティリティ群

セグメンテーション用途のペア画像データセットを整えるためのスクリプト集です。

## 収録ファイル

| ファイル | 役割 |
|----------|------|
| `split_dataset.py` | 画像とマスクを `train/valid/test` に比率分割し、CSV / summary ログを出力 |
| `format_dataset.py` | 分割済みディレクトリを最終提出形式（`train_images/` など）にコピー＆連番リネーム |
| `run_split_and_format.sh` | 上記 2 つをまとめて実行するシェルラッパー |
| `rename_images.py` | 任意ディレクトリ内の画像を連番リネーム（再帰処理対応） |

> 親ディレクトリ側 (`util/datasets/`) の README やスクリプトは後方互換用です。今後は本ディレクトリを直接参照してください。

---

## 必要要件

- Python 3.7 以上（標準ライブラリのみ）
- 画像とマスクは拡張子を除いたベース名が一致していること
- 入力ディレクトリは非再帰スキャン（サブフォルダを含める場合は事前にフラット化、または `rename_images.py --recursive` を利用）

---

## クイックスタート

### 分割のみ

```bash
python3 split_dataset.py \
    --images /path/to/images \
    --masks  /path/to/masks \
    --out    /tmp/dataset_split \
    --train 0.6 --valid 0.2 --test 0.2
```

- 出力ディレクトリ名には自動でタイムスタンプ（`_yyyymmddHHMM`）が付与
- 実行後に `SPLIT_DATASET_OUTPUT_ROOT=/actual/path` が出力される

### 整形のみ

```bash
python3 format_dataset.py \
    --source /tmp/dataset_split_202512171234 \
    --dest   /tmp/dataset_formatted
```

### 分割 + 整形をまとめて

```bash
bash run_split_and_format.sh \
    --images /path/to/images \
    --masks  /path/to/masks \
    --out    /tmp/dataset_split \
    --train 0.6 --valid 0.2 --test 0.2
```

- `--format-dest /path/to/final` で整形出力先を指定可能
- `PYTHON_BIN` 環境変数で Python インタプリタを切り替え可能
- `--dry-run` で分割のドライランのみ実行し整形はスキップ
- 正常完了時は中間ディレクトリを削除、ログを整形済みフォルダに移動

---

## `split_dataset.py` オプション

| オプション | 説明 |
|------------|------|
| `--images` (必須) | 画像ディレクトリ（複数指定可） |
| `--masks` (必須) | マスクディレクトリ（複数指定可） |
| `--out` (必須) | 出力ルート（`train/valid/test` を作成） |
| `--train`, `--valid`, `--test` | 分割比（少数/百分率どちらでも可、自動正規化） |
| `--seed` | 乱数シード（デフォルト 42） |
| `--move` | コピーではなく移動 |
| `--dry-run` | ファイル操作を行わずログのみ生成 |
| `--preserve-test-index N` | 指定インデックスの入力セットを `test` として保持 |
| `--preserve-original-test` | 入力が 2 組のとき index 1 を自動で `test` 保存 |
| `--no-append-timestamp` | 出力ルートへのタイムスタンプ付与を無効化 |

### 出力ログ

- `split_log_<timestamp>.csv` — 各ファイルの割当先・元パスを記録
- `split_summary_<timestamp>.txt` — 総ペア数、train/valid/test 件数

---

## `format_dataset.py` オプション

| オプション | 説明 |
|------------|------|
| `--source` (必須) | `split_dataset.py` の出力ルート |
| `--dest` (必須) | 整形後の出力先（自動作成） |
| `--dry-run` | コピーを行わず実行内容のみ表示 |

### 出力構造

```
<dest>/
├── train_images/     (image1.png, image2.png, ...)
├── train_masks/      (mask1.png, mask2.png, ...)
├── validation_images/
├── validation_masks/
├── test_images/
└── test_masks/
```

---

## `rename_images.py` オプション

| オプション | 説明 |
|------------|------|
| `--dir` (必須) | 対象ディレクトリ |
| `--prefix` | ファイル名の接頭辞（デフォルト `gen_img`） |
| `--start` | 連番の開始番号（デフォルト 0） |
| `--zero-pad` | ゼロ埋め幅 |
| `--auto-pad` | 最大番号に応じて自動ゼロ埋め |
| `--ext` | 対象拡張子（複数指定可、デフォルト `.png`） |
| `--output` | 別ディレクトリにコピー（省略時はインプレースリネーム） |
| `--recursive` | 配下のサブディレクトリを一括処理 |
| `--skip-processed` | マーカーファイルがあればスキップ |
| `--write-marker` | 正常終了後にマーカーを作成 |
| `--marker-name` | マーカーファイル名（デフォルト `.rename_images_done`） |
| `--overwrite` | 既存ファイルの上書きを許可 |
| `--dry-run` | 変更せず計画のみ表示 |

### 使用例

```bash
# 単一ディレクトリ
python3 rename_images.py \
    --dir /datasets/raw \
    --prefix sample \
    --start 1 \
    --zero-pad 4

# 再帰処理
python3 rename_images.py \
    --dir /datasets/raw_assets \
    --recursive \
    --prefix dataset \
    --auto-pad \
    --write-marker
```

- 再帰モードでは「指定拡張子のファイルのみを含むディレクトリ」を自動検出
- `--output` と組み合わせるとルートからの相対パス構造を保持して別ツリーに複製
- 連番は各ディレクトリで独立してリセット
- ファイルは自然順序（0, 1, 2, ..., 9, 10, 11, ...）でソート

---

## `run_split_and_format.sh` 詳細

### 処理フロー

1. `split_dataset.py` を実行
2. `SPLIT_DATASET_OUTPUT_ROOT` を取得
3. `format_dataset.py` を実行
4. ログファイルを整形済みフォルダに移動
5. 中間の split ディレクトリを削除

### 固有オプション

| オプション | 説明 |
|------------|------|
| `--format-dest` | 整形出力先（省略時は `<split出力>_formatted`） |
| `-h`, `--help` | ヘルプ表示 |

その他の引数はすべて `split_dataset.py` にそのまま渡されます。

---

## 補足

- すべて標準ライブラリのみ・依存ゼロ
- Windows / macOS / Linux で同一挙動（`os.path` / `pathlib` ベース）
- 詳細な使い方は親ディレクトリの `util/datasets/README.md` も参照
