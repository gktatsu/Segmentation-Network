# random_split ユーティリティ群

`util/datasets/random_split/` には、セグメンテーション用途のペア画像データセットを以下の手順で整えるためのスクリプトがまとまっています。

1. `split_dataset.py` — 画像とマスクを `train / valid / test` に比率分割し、ログを出力
2. `format_dataset.py` — 分割済みディレクトリを最終提出形式（`train_images/` など）にコピー＆連番リネーム
3. `run_split_and_format.sh` — 上記 1 と 2 をまとめて実行するシェルラッパー
4. `rename_images.py` — 任意ディレクトリ（または再帰的に複数ディレクトリ）の画像を連番リネーム（コピー or インプレース）

> 親ディレクトリ側 (`util/datasets/`) に同名ファイルが残っていますが、そちらは後方互換用の薄いラッパーです。今後は本ディレクトリのスクリプトを直接参照してください。

---

## 必要要件

- Python 3.7 以上（すべて標準ライブラリのみ）
- 画像ファイルとマスクファイルは拡張子を除いたベース名が一致していること
- 入力ディレクトリは非再帰スキャン。サブフォルダを含めたい場合は事前にフラット化するか `rename_images.py` の `--recursive` を利用してください。

---

## クイックスタート

```bash
cd util/datasets
python3 random_split/split_dataset.py \
  --images /path/to/images \
  --masks  /path/to/masks \
  --out    /tmp/dataset_split \
  --train 0.6 --valid 0.2 --test 0.2

python3 random_split/format_dataset.py \
  --source /tmp/dataset_split_202512081230 \
  --dest   /tmp/dataset_formatted
```

- `split_dataset.py` 実行後の標準出力には `SPLIT_DATASET_OUTPUT_ROOT=...` が必ず含まれるので、`format_dataset.py` の `--source` にその値を渡せば OK です。
- `--dry-run` を併用するとファイル操作を行わずに割当ログだけを確認できます。

### まとめて実行

```bash
cd util/datasets
bash random_split/run_split_and_format.sh \
  --images /path/to/images \
  --masks  /path/to/masks \
  --out    /tmp/dataset_split \
  --train 0.6 --valid 0.2 --test 0.2
```

- `--format-dest /path/to/final` を追加するとフォーマット済み出力先を明示的に指定できます。
- `PYTHON_BIN=/path/to/python bash random_split/run_split_and_format.sh ...` のように別の Python 実行ファイルを指定することも可能です。
- `--dry-run` を付けると分割のみをドライランで実行し、整形フェーズはスキップされます。

---

## `split_dataset.py` 主なオプション

| オプション | 説明 |
| --- | --- |
| `--images`, `--masks` | 画像・マスクディレクトリを複数指定可（非再帰スキャン）。|
| `--out` | 分割結果を保存するルート。デフォルトで `_yyyymmddHHMM` が付与されます。|
| `--train/--valid/--test` | 分割比。少数 or 百分率どちらでも可（自動正規化）。|
| `--move` | コピーではなく移動したい場合に指定。|
| `--dry-run` | ログのみ生成。|
| `--preserve-original-test` / `--preserve-test-index` | 既存のテストセットを保持したまま残りを再分配。|
| `--no-append-timestamp` | 出力ルート名へのタイムスタンプ付与を無効化。|

ログとして `split_log_<timestamp>.csv` と `split_summary_<timestamp>.txt` が `--out` 直下に作成されます。

---

## `format_dataset.py`

| オプション | 説明 |
| --- | --- |
| `--source` | `split_dataset.py` の出力ルート（`train/valid/test` 配下に `images/`・`masks/` が必要）。|
| `--dest` | 最終提出フォーマットを書き込む先。存在しなくても自動作成。|
| `--dry-run` | コピーせず予定のみ表示。|

各サブセットについて `image1.png`, `mask1.png` のような連番にリネームされたコピーが生成されます。処理件数は標準出力で確認できます。

---

## `rename_images.py`

```bash
python3 random_split/rename_images.py \
  --dir /datasets/raw_assets \
  --prefix sample \
  --start 0 \
  --zero-pad 4 \
  --ext .png --ext .jpg \
  --output /datasets/renamed \
  --auto-pad --write-marker
```

- デフォルトはインプレースリネーム。`--output` を指定するとコピー先に書き出せます。
- `--recursive` を付けると対象ルート以下で「指定拡張子のみを含むディレクトリ」を自動検出して一括処理します。
- `--skip-processed` + `--write-marker` でマーカーを基準に再実行をスキップ可能。

---

## 補足メモ

- これらのスクリプトはすべて標準ライブラリのみ・依存ゼロです。
- Windows / macOS / Linux で同一挙動になるようパス操作は `os.path`／`pathlib` ベースで記述しています。
- 既存のワークフローで旧パス（`util/datasets/*.py`）を呼び出していても動作しますが、メンテナンス性向上のため順次 `random_split/` 側に切り替えてください。
