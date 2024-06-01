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

# "BASE_OUTPUT": "output",
config_dic = {
    "DATASET_PATH": r"Network/Dataset/Synthetic/",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "PIN_MEMORY": True if "cuda" else False,
    "NUM_CLASSES": 3,
    "NUM_WORKERS": 0,
    "INIT_LR": 0.001,
    "NUM_EPOCHS": 600,
    "BATCH_SIZE": 64,
    "INPUT_IMAGE_WIDTH": 128,
    "INPUT_IMAGE_HEIGHT": 128,
    "THRESHOLD": 0.5,
    "BASE_OUTPUT": "/mnt/hdd/pascalr/Segmentation-Network/output",
    "PATIENCE": 25,
    "MIN_NUM_EPOCHS": 600,
    "NUM_LOG_IMAGES": 5,
    "SCHEDULER_FACTOR": 0.9
}
