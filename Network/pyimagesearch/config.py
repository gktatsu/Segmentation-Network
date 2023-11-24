# import the necessary packages
import torch
import os

config_dic = {
    "DATASET_PATH": r"C:/Users/Platz3/PycharmProjects/Segmentation Network/venv/Network/Dataset/Dataset 1/WannerFIB/",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "PIN_MEMORY": True if "cuda" else False,
    "NUM_CLASSES": 3,
    "INIT_LR": 0.001,
    "NUM_EPOCHS": 40,
    "BATCH_SIZE": 64,
    "INPUT_IMAGE_WIDTH": 128,
    "INPUT_IMAGE_HEIGHT": 128,
    "THRESHOLD": 0.5,
    "BASE_OUTPUT": "output",
    "PATIENCE": 5,
    "MIN_NUM_EPOCHS": 20,
    "NUM_LOG_IMAGES": 5
}
