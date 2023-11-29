# import the necessary packages
import torch
import os

# Epochs: 40, Patience: 5, Min_num_epochs: 20, INIT_LR: 0.001
# "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

# Path for original dataset
# DATASET_PATH": r"C:/Users/Platz3/PycharmProjects/Segmentation Network/venv/Network/Dataset/Dataset 1/WannerFIB/"

# Path for augmented and synthetic datasets
# DATASET_PATH": r"C:/Users/Platz3/PycharmProjects/Segmentation-Network/Network/Dataset/AugDataset/WannerFIB/"

config_dic = {
    "DATASET_PATH": r"Network/Dataset/AugDataset/WannerFIB/",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "PIN_MEMORY": True if "cuda" else False,
    "NUM_CLASSES": 3,
    "INIT_LR": 0.001,
    "NUM_EPOCHS": 500,
    "BATCH_SIZE": 64,
    "INPUT_IMAGE_WIDTH": 128,
    "INPUT_IMAGE_HEIGHT": 128,
    "THRESHOLD": 0.5,
    "BASE_OUTPUT": "output",
    "PATIENCE": 15,
    "MIN_NUM_EPOCHS": 60,
    "NUM_LOG_IMAGES": 5,
    "SCHEDULER_FACTOR": 0.1
}
