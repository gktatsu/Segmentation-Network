# Segmentation-Network

セグメンテーションネットワークは、画像を異なる意味領域（セグメント）に分割するために設計されています。入力画像の各ピクセルにクラスを割り当てるのがこのネットワークの役割です。ネットワークを学習させるには、入力画像とラベル（マスク）のペアが必要です。アーキテクチャは古典的な **U-Net** を採用しています。

> `model.py` — U-Net の全コンポーネントを実装  
> `dataset.py` — 入力データの前処理とオンライン・オーグメンテーションを担当

---

## ディレクトリ構成

```
Segmentation-Network/
├── Network/
│   ├── train.py            # 学習スクリプト
│   ├── Evaluation.py       # 評価スクリプト
│   ├── predict.py          # 推論スクリプト
│   └── pyimagesearch/
│       ├── config.py       # ハイパーパラメータ設定
│       ├── dataset.py      # データセットクラス
│       └── model.py        # U-Net モデル定義
├── util/
│   └── datasets/           # データセット分割・整形ツール群
│       └── random_split/   # 分割ユーティリティ本体
├── run.sh                  # クラスタ実行用スクリプト
├── runEvaluation.sh        # クラスタ評価用スクリプト
└── requirements            # Python 依存ライブラリ
```

---

## 設定（config.py）

`Network/pyimagesearch/config.py` でハイパーパラメータを一元管理します。

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `DATASET_PATH` | `""` | データセットのルートパス |
| `NUM_CLASSES` | `3` | セグメンテーションクラス数 |
| `BATCH_SIZE` | `32` | バッチサイズ |
| `NUM_EPOCHS` | `5000` | 最大エポック数 |
| `INIT_LR` | `0.001` | 初期学習率 |
| `INPUT_IMAGE_WIDTH/HEIGHT` | `512` | 入力画像サイズ |
| `PATIENCE` | `200` | Early stopping の許容エポック数 |
| `MIN_NUM_EPOCHS` | `100` | 早期終了を適用する前の最低エポック数 |
| `SCHEDULER_FACTOR` | `0.1` | 学習率スケジューラの減衰率 |
| `MIN_DELTA` | `1e-4` | 改善と見なす最小ロス減少量 |
| `ONLINE_ROTATION_MAX_DEGREES` | `0.0` | オンライン回転オーグメンテーションの最大角度 |
| `ONLINE_AUGMENTATIONS_PER_IMAGE` | `0` | 回転時の追加画像数 |
| `NUM_LOG_IMAGES` | `5` | WandB にログする画像数 |
| `BASE_OUTPUT` | `/mnt/hdd/.../output/weights/` | 重みの出力先 |

環境変数 (`DATASET_PATH`, `BATCH_SIZE`, `ONLINE_ROTATION_MAX_DEGREES` など) でも上書き可能です。

---

## データセット構造

ネットワークは train / validation / test に分割されたデータセットを期待します。

```
Dataset/
├── train_images/
├── train_masks/
├── validation_images/
├── validation_masks/
├── test_images/
└── test_masks/
```

> 画像とマスクはファイルベース名が一致している必要があります（例: `image1.png` と対応する `mask1.png`）。

データセット準備には `util/datasets/random_split/` のツール群を利用できます。詳細は [util/datasets/README.md](util/datasets/README.md) を参照してください。

---

## ローカル実行

### 学習

```bash
cd Network
python train.py
```

### CLI オプションによるオーバーライド

学習時に主要なパラメータをコマンドラインで上書きできます。

```bash
python train.py \
    --dataset-path /path/to/dataset \
    --base-output /path/to/output \
    --run-name my_experiment \
    --batch-size 64 \
    --num-workers 4 \
    --online-rotation-max-degrees 15 \
    --online-augmentations-per-image 2
```

| オプション | 説明 |
|-----------|------|
| `--dataset-path` | データセットディレクトリのパス |
| `--base-output` | チェックポイント・ログの保存先 |
| `--run-name` | WandB のランネーム（未指定時はタイムスタンプ+UUID） |
| `--batch-size` | バッチサイズ |
| `--num-workers` | DataLoader のワーカー数 |
| `--online-rotation-max-degrees` | オンライン回転の最大角度 (±度) |
| `--online-augmentations-per-image` | 回転オーグメンテーションの複製数 |

### 評価のみ

```bash
python Evaluation.py
```

`Evaluation.py` 内で `model_weights` 変数を適切なパスに変更してください。

---

## クラスタ実行

### 学習

```bash
submit ./run.sh --pytorch --requirements requirements \
    --apt-install libsm6 libxext6 libxrender-dev libglib2.0-0 libgl1-mesa-glx python3-opencv \
    --name training_run
```

### 評価

```bash
submit ./runEvaluation.sh --pytorch --requirements requirements \
    --apt-install libsm6 libxext6 libxrender-dev libglib2.0-0 libgl1-mesa-glx python3-opencv \
    --name evaluation_run
```

> `--pytorch` を省略する場合は `requirements` ファイルに PyTorch 関連パッケージを追加してください。

---

## WandB 統合

学習中は [Weights & Biases](https://wandb.ai/) に以下のメトリクスがログされます。

- `train/loss`, `train/avgTrainLoss`
- `val/avgValLoss`, `val/miou`
- `test/avgTestLoss`, `test/miou`
- 各エポックの画像サンプル

環境変数 `WANDB_API_KEY` または `train.py` 内で API キーを設定してください。

---

## 主な Requirements

- Python >= 3.8
- PyTorch >= 2.1
- TorchVision >= 0.16
- torchmetrics
- scikit-learn
- tqdm
- matplotlib
- numpy
- wandb
- opencv-python (cv2)
- imutils

---

## ライセンス・参考文献

U-Net アーキテクチャは Ronneberger らの論文に基づきます:  
*Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.*