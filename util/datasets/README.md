## 概要

このリポジトリには、ペア画像（例: セグメンテーションの入力画像と対応マスク）を学習・検証・テストに振り分ける `split_dataset.py` と、分割後の成果物を最終的な提出形式に整形する `format_dataset.py` の 2 本の CLI ツールが含まれています。どちらも標準ライブラリのみで動作し、Linux / macOS / Windows いずれの環境でも Python 3.7 以上で利用できます。

## ファイル一覧

| ファイル | 役割 |
|----------|------|
| `split_dataset.py` | 複数の画像ディレクトリとマスクディレクトリを照合し、指定比率で `train` / `valid` / `test` に分割するメインツール。ログ（CSV / summary）を生成します。 |
| `format_dataset.py` | `split_dataset.py` の出力（`train/valid/test` 直下に `images/`・`masks/` を持つ構造）を、`train_images/` などのフラットな最終構造にコピーしつつ連番リネームします。 |

## 必要条件

- Python 3.7 以上
- 画像とマスクで拡張子を除いたベース名が一致していること（例: `img0001.png` と `img0001_mask.png` のような命名では不可）
- 入力ディレクトリは非再帰でスキャンされます。サブフォルダを含めたい場合は事前にフラット化してください。

## 典型的なワークフロー

1. **分割の確認 (dry-run)**
	 ```bash
	 python3 split_dataset.py \
		 --images /path/to/images \
		 --masks  /path/to/masks \
		 --out    /path/to/out_dir \
		 --train 0.6 --valid 0.2 --test 0.2 \
		 --dry-run
	 ```
	 - コピーや移動は行われず、ログのみ生成されます。割当結果を CSV で確認できます。

2. **本実行**
	 ```bash
	 python3 split_dataset.py \
		 --images /path/to/images \
		 --masks  /path/to/masks \
		 --out    /path/to/out_dir \
		 --train 0.6 --valid 0.2 --test 0.2
	 ```
	 - デフォルトではコピー。`--move` を付けると移動になります。

3. **既存の分割を保持しつつ再分配する場合**
	 ```bash
	 python3 split_dataset.py \
		 --images /path/to/train_images /path/to/test_images \
		 --masks  /path/to/train_masks  /path/to/test_masks \
		 --out    /path/to/out_dir \
		 --train 0.6 --valid 0.2 --test 0.2 \
		 --preserve-original-test
	 ```
	 - 2 番目の入力（index 1）のペアはそのまま `test` に残し、残りを `train` / `valid` で再分配します。

4. **最終フォーマットへの整形**
	 ```bash
	 python3 format_dataset.py \
		 --source /path/to/out_dir \
		 --dest   /path/to/final_dataset
	 ```
	 - `final_dataset/` 直下に `train_images/`, `train_masks/`, `validation_images/`, `validation_masks/`, `test_images/`, `test_masks/` が生成され、各サブセットで `image1.png`, `mask1.png` といった連番にコピーされます。

### フォルダ構造イメージ

```
split_out/
	train/
		images/
		masks/
	valid/
		images/
		masks/
	test/
		images/
		masks/

format_dataset.py 実行後:

Dataset/
	train_images/
	train_masks/
	validation_images/
	validation_masks/
	test_images/
	test_masks/
```

## `split_dataset.py` オプション一覧

| オプション | 役割 |
|-------------|------|
| `--images` (必須) | 画像ディレクトリ（複数指定可）。同じベース名でマージします。 |
| `--masks` (必須) | マスクディレクトリ（複数指定可）。 |
| `--out` (必須) | 出力ルート。`train/valid/test` 配下に `images/`・`masks/` を作成します。 |
| `--train`, `--valid`, `--test` | 分割比。少数でも百分率でも指定可能。合計が自動で正規化されます。 |
| `--seed` | 乱数シード。デフォルト 42。 |
| `--move` | コピーではなく移動したい場合に指定。 |
| `--dry-run` | ファイル操作を行わずログのみ生成。 |
| `--preserve-test-index` | 指定したインデックスの入力セットを `test` として保持。 |
| `--preserve-original-test` | 入力が 2 組のときに index 1 を自動で `test` 保存。 |

## `format_dataset.py` オプション一覧

| オプション | 役割 |
|-------------|------|
| `--source` (必須) | `split_dataset.py` の出力ルート。`train/valid/test` 配下に `images/`・`masks/` が必要です。 |
| `--dest` (必須) | 整形後の出力先。存在しない場合は自動作成されます。 |
| `--dry-run` | コピーを行わず、実行内容のみ表示。 |

## 出力されるログ

`split_dataset.py` を実行すると、`--out` 直下にタイムスタンプ付きログが生成されます。

- `split_log_<timestamp>.csv`
	- 列: `basename`, `image_src`, `image_src_dir`, `mask_src`, `mask_src_dir`, `split`, `dest_image`, `dest_mask`
	- スプレッドシートで開けば各ファイルの割当先や元パスを確認できます。
- `split_summary_<timestamp>.txt`
	- 総ペア数、train/valid/test 件数を記録します。

`format_dataset.py` はログファイルを生成しませんが、進捗として各サブセットのコピー件数を標準出力に表示します。

## 実装上の注意

- 重複ベース名が複数の入力ディレクトリで見つかった場合は安全のためエラーで停止します。
- 余剰サンプルはすべて `train` に割り当てています。別ルールが必要であれば `compute_counts` を変更してください。
- `--preserve-test-index` を使うと、指定ディレクトリのサンプルが `test` に固定され、残りのデータで比率を再計算します。
- `format_dataset.py` はベース名が一致しないペアを自動的にスキップし、共通部分のみ整形します。

## トラブルシューティング

| 症状 | 原因・対処 |
|------|------------|
| `No paired files found` | ベース名が一致していない、または画像・マスクどちらかしか存在しない。命名規則と入力パスを確認してください。 |
| `Duplicate basenames detected` | 複数の入力ディレクトリに同名ペアが存在。対象ファイルの整理が必要です。 |
| ログはあるがファイルがコピーされない | `--dry-run` を付けたまま実行している可能性があります。フラグを外してください。 |
| 目的の分割比にならない | サンプル数が少ない場合は丸めによって偏りが出ます。`compute_counts` のロジックを調整するか、サンプル数を増やしてください。 |

## 今後の拡張アイデア

- 再帰的なサブフォルダ探索
- 余剰サンプルのランダム再配分
- コピー完了後のハッシュ整合性チェック
- 進捗バー（`tqdm`）や並列コピーの導入

---

ご自身の環境に合わせた具体的なコマンド例や、追加機能の要望があれば遠慮なく共有してください。必要に応じて README やスクリプトをさらに拡充できます。

