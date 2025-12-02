# import the necessary packages
import torch
import os

# Epochs: 40, Patience: 5, Min_num_epochs: 20, INIT_LR: 0.001, Augmentation (XXL_mix) LR: 0.002875, Batch_size: 512
# "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

# Path for original dataset
# DATASET_PATH": r"C:/Users/Platz3/PycharmProjects/Segmentation Network/venv/Network/Dataset/Dataset 1/WannerFIB/"

# Path for augmented and synthetic datasets
# DATASET_PATH": r"C:/Users/Platz3/PycharmProjects/Segmentation-Network/Network/Dataset/AugDataset/WannerFIB/"

# Path for Cluster:
# "DATASET_PATH": r"Network/Dataset/AugDataset/WannerFIB/"

# ========== Add by Tatsuki ==========
# Path for Demo:
# "DATASET_PATH": r"/mnt/hdd/tatsuki/tatsuki/datasets/Segmentation-Network/demo/"

# Path for Dataset 1:
# "DATASET_PATH": r"/mnt/hdd/tatsuki/tatsuki/datasets/Segmentation-Network/original/Dataset 1/WannerFIB_202510271742"

# "BASE_OUTPUT": "output",
config_dic = {
    "DATASET_PATH": r"",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "PIN_MEMORY": True if "cuda" else False,
    "NUM_CLASSES": 3,
    "NUM_WORKERS": 0,
    "INIT_LR": 0.001,
    "NUM_EPOCHS": 5000,
    "BATCH_SIZE": 32,
    "INPUT_IMAGE_WIDTH": 512,
    "INPUT_IMAGE_HEIGHT": 512,
    "THRESHOLD": 0.5,
    "BASE_OUTPUT": "/mnt/hdd/tatsuki/tatsuki/programs/Segmentation-Network/Network/output/weights/",
    "PATIENCE": 200,  # early stopping patience
    "MIN_NUM_EPOCHS": 100,
    "NUM_LOG_IMAGES": 5,
    "SCHEDULER_FACTOR": 0.1,
    "MIN_DELTA": 1e-4,
    # online augmentation controls
    "ONLINE_ROTATION_MAX_DEGREES": 0.0,
    "ONLINE_AUGMENTATIONS_PER_IMAGE": 0
}


def _coerce_value(key, value):
    """Coerce override values to match the original config types."""
    if key not in config_dic:
        return value

    template = config_dic[key]
    if isinstance(template, bool):
        if isinstance(value, bool):
            return value
        return str(value).lower() in {"1", "true", "yes", "on"}
    if isinstance(template, int) and not isinstance(template, bool):
        return int(value)
    if isinstance(template, float):
        return float(value)
    return value


def apply_overrides(**overrides):
    """Override config values at runtime (e.g., via CLI arguments)."""
    for key, value in overrides.items():
        if value is None:
            continue
        config_dic[key] = _coerce_value(key, value)


def _apply_env_overrides():
    env_map = {
        "DATASET_PATH": "DATASET_PATH",
        "BASE_OUTPUT": "BASE_OUTPUT",
        "BATCH_SIZE": "BATCH_SIZE",
        "NUM_WORKERS": "NUM_WORKERS",
    }
    overrides = {}
    for cfg_key, env_key in env_map.items():
        env_val = os.environ.get(env_key)
        if env_val is not None:
            overrides[cfg_key] = env_val
    if overrides:
        apply_overrides(**overrides)


_apply_env_overrides()
