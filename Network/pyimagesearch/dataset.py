# import the necessary packages
from torch.utils.data import Dataset
from pyimagesearch import config
import numpy as np
import cv2
import glob
import torch
import matplotlib.pyplot as plt
import random

class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.imagePaths = glob.glob(imagePaths+"/*.png")
		self.maskPaths = glob.glob(maskPaths+"/*.png")
		#import pdb
		#pdb.set_trace()
		self.transforms = transforms
	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.imagePaths)
	def __getitem__(self, idx):
		probability = random.random() < 0.6

		if probability:
			# use original image
			imagePath = self.imagePaths[idx]
			image = cv2.imread(imagePath)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			mask = cv2.imread(self.maskPaths[idx], 0)
		else:
			# use augmented image
			base_image_name = self.imagePaths[idx]
			augmentations = ["blur_0", "brightness_0", "crop_0", "noise_0", "zoom_rotate_0"]
			chosen_augmentation = random.choice(augmentations)
			augmented_image_path = f"{base_image_name}_aug_{chosen_augmentation}.png"

			image = cv2.imread(augmented_image_path)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			mask = cv2.imread(augmented_image_path, 0)

		binary_mask = np.zeros((mask.shape[0], mask.shape[1], config.config_dic["NUM_CLASSES"]), dtype=np.uint8)

		for class_idx in range(config.config_dic["NUM_CLASSES"]):
			binary_mask[:, :, class_idx] = (mask == class_idx).astype(np.uint8)

		# apply the transformations to both image and its mask
		if self.transforms is not None:
			image = self.transforms(image)
			mask = self.transforms(binary_mask)
			mask = (mask * 255)

		return (image, mask)
