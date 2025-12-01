# import the necessary packages
from torch.utils.data import Dataset
from pyimagesearch import config
import numpy as np
import cv2
import glob
import torch
import random


class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms,
                 rotation_degrees=0.0, augmentations_per_image=0):
        # store the image and mask filepaths, and augmentation
        # transforms
        self.imagePaths = glob.glob(imagePaths+"/*.png")
        self.maskPaths = glob.glob(maskPaths+"/*.png")
        self.transforms = transforms
        self.rotation_degrees = float(rotation_degrees)
        self.augmentations_per_image = max(0, int(augmentations_per_image))
        self.num_classes = config.config_dic["NUM_CLASSES"]
        self.base_length = len(self.imagePaths)

        if len(self.imagePaths) != len(self.maskPaths):
            raise ValueError(
                "The number of images and masks must match for augmentation."
            )

        if self.rotation_degrees <= 0:
            self.total_multiplier = 1
        else:
            self.total_multiplier = 1 + self.augmentations_per_image

    def __len__(self):
        # return the number of total samples contained in the dataset
        return self.base_length * self.total_multiplier

    def __getitem__(self, idx):
        # grab the image path from the current index
        base_index = idx % self.base_length
        imagePath = self.imagePaths[base_index]
        # load the image from disk, swap its channels from BGR to RGB,
        # and read the associated mask from disk in grayscale mode
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.maskPaths[base_index], 0)

        if self._should_apply_rotation(idx):
            image, mask = self._apply_rotation(image, mask)

        binary_mask = np.zeros(
            (mask.shape[0], mask.shape[1], self.num_classes), dtype=np.uint8)

        for class_idx in range(self.num_classes):
            binary_mask[:, :, class_idx] = (mask == class_idx).astype(np.uint8)

        """"""
        # check to see if we are applying any transformations

        if self.transforms is not None:
            python_state = random.getstate()
            torch_state = torch.random.get_rng_state()
            numpy_state = np.random.get_state()

            image = self.transforms(image)

            random.setstate(python_state)
            torch.random.set_rng_state(torch_state)
            np.random.set_state(numpy_state)

            mask = self.transforms(binary_mask)
            # mask channels are one-hot; convert to per-pixel class labels
            if isinstance(mask, torch.Tensor):
                # if mask is float tensor in [0,1], argmax works directly
                mask = mask.argmax(dim=0).type(torch.LongTensor)
            else:
                # fallback: convert to numpy then argmax
                mask = torch.from_numpy(np.argmax(mask, axis=0)).long()
        # import pdb
        # pdb.set_trace

    # image.shape sollte sein: C,B,H
    # mask.shape num_classes,B,H  oder B,H
    # return a tuple of the image and its mask
        return (image, mask)

    def _should_apply_rotation(self, global_index):
        if self.rotation_degrees <= 0:
            return False
        if self.augmentations_per_image <= 0:
            return False
        if self.base_length == 0:
            return False
        augmentation_round = global_index // self.base_length
        return augmentation_round > 0

    def _apply_rotation(self, image, mask):
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        (h, w) = image.shape[:2]
        center = (w / 2, h / 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_image = cv2.warpAffine(
            image,
            rot_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        rotated_mask = cv2.warpAffine(
            mask,
            rot_matrix,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        rotated_mask = rotated_mask.astype(mask.dtype)

        return rotated_image, rotated_mask
