# USAGE
# python train.py
# import the necessary packages
from pyimagesearch.dataset import SegmentationDataset
from pyimagesearch.model import UNet
from pyimagesearch import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
import wandb
from torchmetrics import JaccardIndex
from torch.optim.lr_scheduler import ReduceLROnPlateau

# wandb.login(key="a62bd616c3a898497ab242a339258e281c14489e")
# os.environ["WANDB_MODE"] = "dryrun"

# start a new wandb run to track this script
wandb.init(
	# set the wandb project where this run will be logged
    project="Abgabe",
    # track hyperparameters and run metadata
    config=config.config_dic
)

# adapt path to model weights

# 232 images
# real images
# model_weights = r"Network/Weights (Abgabe)/Dataset 1/Run 1/"
# model_weights = r"Network/Weights (Abgabe)/Dataset 1/Run 2/"
# model_weights = r"Network/Weights (Abgabe)/Dataset 1/Run 3/" # Leer
# synthetic
# model_weights = r"Network/Weights (Abgabe)/Synthetic/Run 1/"
model_weights = r"Network/Weights (Abgabe)/Synthetic/Run 2/"
# model_weights = r"Network/Weights (Abgabe)/Synthetic/Run 3/"

# 464 images
# ControlNet augmentations
# model_weights = r"Network/Weights (Abgabe)/Synthetic + Real (small)/Run 1/"
# model_weights = r"Network/Weights (Abgabe)/Synthetic + Real (small)/Run 2/"
# model_weights = r"Network/Weights (Abgabe)/Synthetic + Real (small)/Run 3/"
# standard augmentations
# model_weights = r"Network/Weights (Abgabe)/Augmentations (small)/Run 1/"
# model_weights = r"Network/Weights (Abgabe)/Augmentations (small)/Run 2/"
# model_weights = r"Network/Weights (Abgabe)/Augmentations (small)/Run 3/"

# 5393 images
# ControlNet augmentations
# model_weights = r"Network/Weights (Abgabe)/Synthetic + Real (medium)/Run 1/"
# model_weights = r"Network/Weights (Abgabe)/Synthetic + Real (medium)/Run 2/"
# model_weights = r"Network/Weights (Abgabe)/Synthetic + Real (medium)/Run 3/"
# standard augmentations
# model_weights = r"Network/Weights (Abgabe)/Augmentations (medium)/Run 1/"
# model_weights = r"Network/Weights (Abgabe)/Augmentations (medium)/Run 2/"
# model_weights = r"Network/Weights (Abgabe)/Augmentations (medium)/Run 3/"

# 10.553 images
# ControlNet augmentations
# model_weights = r"Network/Weights (Abgabe)/Synthetic + Real (large)/Run 1/"
# model_weights = r"Network/Weights (Abgabe)/Synthetic + Real (large)/Run 2/"
# model_weights = r"Network/Weights (Abgabe)/Synthetic + Real (large)/Run 3/"
# standard augmentations
# model_weights = r"Network/Weights (Abgabe)/Augmentations (large)/Run 1/"
# model_weights = r"Network/Weights (Abgabe)/Augmentations (large)/Run 2/"
# model_weights = r"Network/Weights (Abgabe)/Augmentations (large)/Run 3/"

# time match: ControlNet augmentations (10.553) <-> standard augmentations (31.560)
# model_weights = r"Network/Weights (Abgabe)/Augmentations (Time match)/Run 1/"
# model_weights = r"Network/Weights (Abgabe)/Augmentations (Time match)/Run 2/"
# model_weights = r"Network/Weights (Abgabe)/Augmentations (Time match)/Run 3/"

testImages = os.path.join(config.config_dic["DATASET_PATH"], "test_images")
testMasks = os.path.join(config.config_dic["DATASET_PATH"], "test_masks")

# define transformations
transforms = transforms.Compose([transforms.ToPILImage(),
 	transforms.Resize((config.config_dic["INPUT_IMAGE_HEIGHT"],
		config.config_dic["INPUT_IMAGE_WIDTH"])),
	transforms.ToTensor()])

testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
    transforms=transforms)
print(f"[INFO] found {len(testDS)} examples in the test set...")

testLoader = DataLoader(testDS, shuffle=False,
	batch_size=config.config_dic["BATCH_SIZE"], pin_memory=config.config_dic["PIN_MEMORY"],
	num_workers=config.config_dic['NUM_WORKERS'])

sigmoid = torch.nn.Sigmoid()
jaccard = JaccardIndex(task='multilabel', num_labels=config.config_dic["NUM_CLASSES"],threshold = config.config_dic["THRESHOLD"]).to(config.config_dic['DEVICE'])

# initialize our UNet model
unet = UNet(nbClasses=config.config_dic["NUM_CLASSES"]).to(config.config_dic["DEVICE"])

# initialize loss function and optimizer
lossFunc = BCEWithLogitsLoss()
# calculate steps per epoch for training and test set
testSteps = np.max([len(testDS) // config.config_dic["BATCH_SIZE"], 1])

# loop over epochs
print("[INFO] eval the network...")
startTime = time.time()
totalTestLoss = 0
# Load weights for testing
if os.path.exists(model_weights + 'model.pth'):
    print("[INFO] Loading pre-trained weights for testing...")
    unet.load_state_dict(torch.load(model_weights + 'model.pth'))
else:
    print("[WARNING] Weights not found.")

jaccard.reset()

with torch.no_grad():
	# set the model in evaluation mode
	unet.eval()
	# loop over the test set
	for testIndex,(x, y) in enumerate(testLoader):
		# send the input to the device
		(x, y) = (x.to(config.config_dic["DEVICE"]), y.to(config.config_dic["DEVICE"]))
		# make the predictions and calculate the validation loss
		pred = unet(x)
		totalTestLoss += lossFunc(pred, y)
		jaccard(sigmoid(pred),y.long())

		if(testIndex == 0):
			num_img = np.min((x.shape[0],config.config_dic["NUM_LOG_IMAGES"]))
			sigmoid_pediction = sigmoid(pred).cpu()
			x_cpu = x.cpu()
			y_cpu = y.cpu()
			for i in range(num_img):
				fig,axs = plt.subplots(1,3)
				axs[0].imshow(x_cpu[i].permute(1,2,0))
				axs[1].imshow(y_cpu[i].permute(1,2,0))
				axs[2].imshow(sigmoid_pediction[i].permute(1,2,0))
				# axs[3].imshow((sigmoid_pediction[i] > config.config_dic["THRESHOLD"]).float().permute(1, 2, 0))
				for a in axs:
					a.set_axis_off()
				plt.tight_layout()
				wandb.log({f"testImage {i}": wandb.Image(plt)})
				plt.close()

avgTestLoss = totalTestLoss / testSteps
wandb.log({"test/avgTestLoss": avgTestLoss})
miou = jaccard.compute()
wandb.log({"test/miou": miou})

# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

# wandb.sync()
wandb.finish()
