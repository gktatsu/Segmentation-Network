# import the necessary packages
from torch.utils.data import Dataset
from pyimagesearch import config
import numpy as np
import cv2
import glob
import torch
import matplotlib.pyplot as plt
import random
import os

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
		print(f"[Probability] is {probability}. Choose image now...")

		if probability:
			print(f"[Real] Take real image.")
			paths = config.config_dic["DATASET_PATH"] + "/train_images/"

			while True:
				random_image = random.randint(1, 322)
				image_path = os.path.join(paths, f"img_{random_image}")

				if os.path.exists(image_path + ".png"):
					print(f"[Image] {image_path}.png")
					break

			self.imagePaths[idx] = glob.glob(os.path.join(paths, f"{random_image}")+"/*.png")
			# self.maskPaths[idx] = glob.glob(self.imagePaths[idx].replace("train_images", "train_masks")+"/*.png")
			print(f"[Image] is real image {self.imagePaths[idx]}.")
			# print(f"[Mask] is real mask{self.maskPaths[idx]}.")

		else:
			print(f"[Augmented] Take augmented image.")
			paths = config.config_dic["DATASET_PATH"] + "/train_images/"

			while True:
				random_image = random.randint(1, 322)
				image_path = os.path.join(paths, f"img_{random_image}")

				if os.path.exists(image_path + ".png"):
					print(f"[Augmented image] {image_path}.png")
					break

			augmentations = ["blur_0", "brightness_0", "noise_0", "zoom_rotate_0", "zoom_rotate_1"]
			chosen_augmentation = random.choice(augmentations)
			print(f"[Chosen augmentation] {chosen_augmentation}.")
			self.imagePaths[idx] = glob.glob(os.path.join(paths, f"{random_image}_aug_{chosen_augmentation}")+"/*.png")
			# self.maskPaths[idx] = glob.glob(self.imagePaths[idx].replace("train_images", "train_masks")+"/*.png")
			print(f"[Image] is augmented image {self.imagePaths[idx]}.")
			# print(f"[Mask] is augmented mask{self.maskPaths[idx]}.")

		# grab the image path from the current index
		imagePath = self.imagePaths[idx]
		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
		image = cv2.imread(imagePath)
		"""import pdb
		pdb.set_trace()"""
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(self.maskPaths[idx], 0)


		binary_mask = np.zeros((mask.shape[0], mask.shape[1], config.config_dic["NUM_CLASSES"]), dtype=np.uint8)

		for class_idx in range(config.config_dic["NUM_CLASSES"]):
			binary_mask[:,:,class_idx] = (mask==class_idx).astype(np.uint8)

		""""""
		# check to see if we are applying any transformations

		if self.transforms is not None:
			# apply the transformations to both image and its mask
			image = self.transforms(image)
			mask = self.transforms(binary_mask)
			mask = (mask*255) #.type(torch.IntTensor)
		"""fig, axs = plt.subplots(1,config.NUM_CLASSES)
		for i in range(config.NUM_CLASSES):
			axs[i].imshow(mask[i])
		plt.show()"""

		#import pdb
		#pdb.set_trace


			# image.shape sollte sein: C,B,H
			# mask.shape num_classes,B,H  oder B,H
		# return a tuple of the image and its mask
		return (image, mask)
