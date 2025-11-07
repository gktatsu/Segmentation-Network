# import the necessary packages
from torch.utils.data import Dataset
from pyimagesearch import config
import numpy as np
import cv2
import glob
import torch
import matplotlib.pyplot as plt


class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.imagePaths = glob.glob(imagePaths+"/*.png")
        self.maskPaths = glob.glob(maskPaths+"/*.png")
        # import pdb
        # pdb.set_trace()
        self.transforms = transforms

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # grab the image path from the current index
        imagePath = self.imagePaths[idx]
        # load the image from disk, swap its channels from BGR to RGB,
        # and read the associated mask from disk in grayscale mode
        image = cv2.imread(imagePath)
        """import pdb
		pdb.set_trace()"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.maskPaths[idx], 0)

        binary_mask = np.zeros(
            (mask.shape[0], mask.shape[1], config.config_dic["NUM_CLASSES"]), dtype=np.uint8)

        for class_idx in range(config.config_dic["NUM_CLASSES"]):
            binary_mask[:, :, class_idx] = (mask == class_idx).astype(np.uint8)

        """"""
        # check to see if we are applying any transformations

        if self.transforms is not None:
            # apply the transformations to both image and its mask
            image = self.transforms(image)
            # transforms will convert to tensor with shape (C, H, W)
            mask = self.transforms(binary_mask)
            # mask channels are one-hot encoded per class; convert to class labels
            # multiply by 255 to preserve original 0/255 convention if any, then argmax over channels
            # resulting mask will have shape (H, W) with values in {0,..,NUM_CLASSES-1}
            if isinstance(mask, torch.Tensor):
                # if mask is float tensor in [0,1], argmax works directly
                mask = mask.argmax(dim=0).type(torch.LongTensor)
            else:
                # fallback: convert to numpy then argmax
                mask = torch.from_numpy(np.argmax(mask, axis=0)).long()
        """fig, axs = plt.subplots(1,config.NUM_CLASSES)
		for i in range(config.NUM_CLASSES):
			axs[i].imshow(mask[i])
		plt.show()"""

        # import pdb
        # pdb.set_trace

        # image.shape sollte sein: C,B,H
        # mask.shape num_classes,B,H  oder B,H
        # return a tuple of the image and its mask (image: Tensor CxHxW, mask: LongTensor HxW)
        return (image, mask)
