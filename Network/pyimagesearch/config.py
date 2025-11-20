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
    "DATASET_PATH": r"/mnt/hdd/tatsuki/tatsuki/datasets/Segmentation-Network/original/Dataset 1/WannerFIB_202511201005",
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
    "MIN_DELTA": 1e-4
}
