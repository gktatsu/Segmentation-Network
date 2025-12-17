# Segmentation-Network

The Segmentation Network is designed to partition images into different semantic regions (segments). The network assigns a class to each pixel in the input image. To train the network, pairs of input images and labels (masks) are required. The architecture is based on the classic **U-Net**.

> `model.py` — Implements all U-Net components  
> `dataset.py` — Handles input data preprocessing and online augmentation

---

## Directory Structure

```
Segmentation-Network/
├── Network/
│   ├── train.py            # Training script
│   ├── Evaluation.py       # Evaluation script
│   ├── predict.py          # Inference script
│   └── pyimagesearch/
│       ├── config.py       # Hyperparameter configuration
│       ├── dataset.py      # Dataset class
│       └── model.py        # U-Net model definition
├── util/
│   └── datasets/           # Dataset splitting & formatting tools
│       └── random_split/   # Splitting utilities
├── run.sh                  # Cluster execution script
├── runEvaluation.sh        # Cluster evaluation script
└── requirements            # Python dependencies
```

---

## Configuration (config.py)

Hyperparameters are centrally managed in `Network/pyimagesearch/config.py`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATASET_PATH` | `""` | Root path to the dataset |
| `NUM_CLASSES` | `3` | Number of segmentation classes |
| `BATCH_SIZE` | `32` | Batch size |
| `NUM_EPOCHS` | `5000` | Maximum number of epochs |
| `INIT_LR` | `0.001` | Initial learning rate |
| `INPUT_IMAGE_WIDTH/HEIGHT` | `512` | Input image size |
| `PATIENCE` | `200` | Early stopping patience (epochs) |
| `MIN_NUM_EPOCHS` | `100` | Minimum epochs before early stopping |
| `SCHEDULER_FACTOR` | `0.1` | Learning rate scheduler decay factor |
| `MIN_DELTA` | `1e-4` | Minimum loss decrease to be considered an improvement |
| `ONLINE_ROTATION_MAX_DEGREES` | `0.0` | Maximum angle for online rotation augmentation |
| `ONLINE_AUGMENTATIONS_PER_IMAGE` | `0` | Number of additional rotated copies per image |
| `NUM_LOG_IMAGES` | `5` | Number of images to log to WandB |
| `BASE_OUTPUT` | `/mnt/hdd/.../output/weights/` | Output directory for weights |

These can also be overridden via environment variables (`DATASET_PATH`, `BATCH_SIZE`, `ONLINE_ROTATION_MAX_DEGREES`, etc.).

---

## Dataset Structure

The network expects a dataset split into train / validation / test.

```
Dataset/
├── train_images/
├── train_masks/
├── validation_images/
├── validation_masks/
├── test_images/
└── test_masks/
```

> Image and mask files must have matching base names (e.g., `image1.png` with corresponding `mask1.png`).

For dataset preparation, you can use the tools in `util/datasets/random_split/`. See [util/datasets/README.md](util/datasets/README.md) for details.

---

## Local Execution

### Training

```bash
cd Network
python train.py
```

### CLI Option Overrides

You can override key parameters via command line during training.

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

| Option | Description |
|--------|-------------|
| `--dataset-path` | Path to dataset directory |
| `--base-output` | Directory for checkpoints and logs |
| `--run-name` | WandB run name (defaults to timestamp+UUID) |
| `--batch-size` | Batch size |
| `--num-workers` | Number of DataLoader workers |
| `--online-rotation-max-degrees` | Maximum online rotation angle (±degrees) |
| `--online-augmentations-per-image` | Number of rotated augmentation copies |

### Evaluation Only

```bash
python Evaluation.py
```

Update the `model_weights` variable in `Evaluation.py` to the appropriate path.

---

## Cluster Execution

### Training

```bash
submit ./run.sh --pytorch --requirements requirements \
    --apt-install libsm6 libxext6 libxrender-dev libglib2.0-0 libgl1-mesa-glx python3-opencv \
    --name training_run
```

### Evaluation

```bash
submit ./runEvaluation.sh --pytorch --requirements requirements \
    --apt-install libsm6 libxext6 libxrender-dev libglib2.0-0 libgl1-mesa-glx python3-opencv \
    --name evaluation_run
```

> If you omit `--pytorch`, add PyTorch-related packages to the `requirements` file.

---

## WandB Integration

During training, the following metrics are logged to [Weights & Biases](https://wandb.ai/):

- `train/loss`, `train/avgTrainLoss`
- `val/avgValLoss`, `val/miou`
- `test/avgTestLoss`, `test/miou`
- Image samples per epoch

### API Key Setup (Required)

The `WANDB_API_KEY` environment variable **must** be set before running training:

```bash
export WANDB_API_KEY='your-api-key'
```

To obtain your API key, visit https://wandb.ai/authorize.

If the environment variable is not set, `train.py` will exit with an error message.

---

## Main Requirements

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

## License & References

The U-Net architecture is based on the paper by Ronneberger et al.:  
*Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.*
