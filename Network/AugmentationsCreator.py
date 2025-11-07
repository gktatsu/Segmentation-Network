import os
from torchvision import transforms
from PIL import Image
import random
import torch

# Define the paths to the original images and masks
image_folder = "train_images_real"
mask_folder = "train_masks_real"

# Create the output folders for the augmented images and masks
output_image_folder = "Dataset 1 Augmented (small) - images"
output_mask_folder = "Dataset 1 Augmented (small) - masks"
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_mask_folder, exist_ok=True)

# Define the transformations for augmentation


def get_random_transform():
    return transforms.Compose([
        # transforms.RandomVerticalFlip(),
        # transforms.RandomHorizontalFlip(),
        # transforms.ColorJitter(contrast=0.2, saturation=0.2),
        transforms.RandomRotation(degrees=8),
        # transforms.ColorJitter(brightness=0.2),
        # transforms.RandomCrop(512),
    ])


# List all filenames in the original images folder
image_files = os.listdir(image_folder)

# Number of desired augmentations per image
num_augmentations = 1

# Iterate over each original image and apply the transformations
for image_file in image_files:
    # Path to the original image and mask
    image_path = os.path.join(image_folder, image_file)
    mask_path = os.path.join(mask_folder, image_file)

    # Load the image and the mask
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # Apply a random number of augmentations
    for idx in range(num_augmentations):
        random_transform = get_random_transform()

        # Apply the same transformations to both image and mask
        seed = torch.seed()
        torch.manual_seed(seed)
        augmented_image = random_transform(image)
        torch.manual_seed(seed)
        augmented_mask = random_transform(mask)

        # Convert the augmented images and masks to tensors
        to_tensor = transforms.ToTensor()
        augmented_image = to_tensor(augmented_image)
        augmented_mask = to_tensor(augmented_mask)

        # Save the augmented images and masks
        output_image_path = os.path.join(
            output_image_folder, f"{image_file[:-4]}_{idx}.png")
        output_mask_path = os.path.join(
            output_mask_folder, f"{image_file[:-4]}_{idx}.png")

        # Convert tensor objects to PIL Images for saving
        augmented_image_pil = transforms.ToPILImage()(augmented_image)
        augmented_mask_pil = transforms.ToPILImage()(augmented_mask)

        augmented_image_pil.save(output_image_path)
        augmented_mask_pil.save(output_mask_path)

        print(f"Image {output_image_path} saved!")
        print(f"Mask {output_mask_path} saved!")

print("Augmentation completed.")
