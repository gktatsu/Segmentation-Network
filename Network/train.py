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
    project="Segmentation Network",
    # track hyperparameters and run metadata
    config=config.config_dic
)

logging_path = config.config_dic["BASE_OUTPUT"] + "/" + str(wandb.run.name) + "/"
os.makedirs(logging_path,exist_ok=True)

trainImages = os.path.join(config.config_dic["DATASET_PATH"], "train_images")
trainMasks = os.path.join(config.config_dic["DATASET_PATH"], "train_masks")
valImages = os.path.join(config.config_dic["DATASET_PATH"], "validation_images")
valMasks = os.path.join(config.config_dic["DATASET_PATH"], "validation_masks")
testImages = os.path.join(config.config_dic["DATASET_PATH"], "test_images")
testMasks = os.path.join(config.config_dic["DATASET_PATH"], "test_masks")

# define transformations
transforms = transforms.Compose([transforms.ToPILImage(),
 	transforms.Resize((config.config_dic["INPUT_IMAGE_HEIGHT"],
		config.config_dic["INPUT_IMAGE_WIDTH"])),
	transforms.ToTensor()])

"""transforms = transforms.Compose([transforms.Resize((config.INPUT_IMAGE_HEIGHT,
		config.INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()])"""
# create the train and test datasets
trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
	transforms=transforms)
valDS = SegmentationDataset(imagePaths=valImages, maskPaths=valMasks,
    transforms=transforms)
testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
    transforms=transforms)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(valDS)} examples in the validation set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")

# create the training and test data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=config.config_dic["BATCH_SIZE"], pin_memory=config.config_dic["PIN_MEMORY"],
	num_workers=config.config_dic['NUM_WORKERS'])
valLoader = DataLoader(valDS, shuffle=False,
	batch_size=config.config_dic["BATCH_SIZE"], pin_memory=config.config_dic["PIN_MEMORY"],
	num_workers=config.config_dic['NUM_WORKERS'])
testLoader = DataLoader(testDS, shuffle=False,
	batch_size=config.config_dic["BATCH_SIZE"], pin_memory=config.config_dic["PIN_MEMORY"],
	num_workers=config.config_dic['NUM_WORKERS'])

sigmoid = torch.nn.Sigmoid()
# jaccard = JaccardIndex(task='multiclass', num_classes=config.config_dic["NUM_CLASSES"],threshold = config.config_dic["THRESHOLD"]).to(config.config_dic["DEVICE"])
jaccard = JaccardIndex(task='multiclass', num_classes=config.config_dic["NUM_CLASSES"],threshold = config.config_dic["THRESHOLD"]).to(config.config_dic['DEVICE'])

# initialize our UNet model
unet = UNet(nbClasses=config.config_dic["NUM_CLASSES"]).to(config.config_dic["DEVICE"])

# initialize loss function and optimizer
lossFunc = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr=config.config_dic["INIT_LR"])
# calculate steps per epoch for training and test set
trainSteps = np.max([len(trainDS) // config.config_dic["BATCH_SIZE"], 1]) # len(trainLoader)
valSteps = np.max([len(valDS) // config.config_dic["BATCH_SIZE"], 1]) # 25 // 64
testSteps = np.max([len(testDS) // config.config_dic["BATCH_SIZE"], 1])

# initialize a dictionary to store training history
H = {"train_loss": [], "val_loss": [] , "test_loss": []}

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
currentPatience = 0
bestValLoss = np.inf

scheduler = ReduceLROnPlateau(opt, mode='min', patience=config.config_dic["PATIENCE"], factor=config.config_dic["SCHEDULER_FACTOR"], verbose=True)

for e in tqdm(range(config.config_dic["NUM_EPOCHS"])):
	# set the model in training mode
	unet.train()
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalTestLoss = 0
	totalValLoss = 0

	# loop over the training set
	for (i, (x, y)) in enumerate(trainLoader):
		# send the input to the device
		(x, y) = (x.to(config.config_dic["DEVICE"]), y.to(config.config_dic["DEVICE"]))
		# perform a forward pass and calculate the training loss
		pred = unet(x)
		# import pdb
		# pdb.set_trace()
		loss = lossFunc(pred, y)
		# first, zero out any previously accumulated gradients, then
		# perform backpropagation, and then update model parameters
		opt.zero_grad()
		loss.backward()
		opt.step()
		# add the loss to the total training loss so far
		totalTrainLoss += loss

		wandb.log({"train/loss": loss})

	# switch off autograd for validation
	with torch.no_grad():
		# set the model in evaluation mode
		unet.eval()
		# loop over the validation set
		for valIndex,(x, y) in enumerate(valLoader):
			# send the input to the device
			(x, y) = (x.to(config.config_dic["DEVICE"]), y.to(config.config_dic["DEVICE"]))
			# make the predictions and calculate the validation loss
			# import pdb
			# pdb.set_trace()
			pred = unet(x)
			totalValLoss += lossFunc(pred, y)
			jaccard(sigmoid(pred),y)
			if(valIndex == 0):
				num_img = np.min((x.shape[0],config.config_dic["NUM_LOG_IMAGES"]))
				sigmoid_pediction = sigmoid(pred).cpu()
				x_cpu = x.cpu()
				y_cpu = y.cpu()
				for i in range(num_img):
					#import pdb
					#pdb.set_trace()
					fig,axs = plt.subplots(1,3)
					axs[0].imshow(x_cpu[i].permute(1,2,0))
					axs[1].imshow(y_cpu[i].permute(1,2,0))
					axs[2].imshow(sigmoid_pediction[i].permute(1,2,0))
					for a in axs:
						a.set_axis_off()
					plt.tight_layout()
					wandb.log({f"validationImage {i}": wandb.Image(plt)})
					plt.close()

	# calculate the average training and validation loss
	#print("valSteps: ", valSteps)
	#print("totalTrainLoss:", totalTrainLoss)
	#print("totalValLoss", totalValLoss)
	avgTrainLoss = totalTrainLoss / trainSteps
	avgValLoss = totalValLoss / valSteps

	miou = jaccard.compute()

	wandb.log({"train/avgTrainLoss": avgTrainLoss})
	wandb.log({"val/avgValLoss": avgValLoss})

	wandb.log({"val/miou": miou})
	jaccard.reset()

	if(avgValLoss < bestValLoss):
		bestValLoss = avgValLoss
		#import pdb
		#pdb.set_trace()
		torch.save(unet.state_dict(),logging_path + "model.pth")
		#torch.save(unet, config.MODEL_PATH)
		currentPatience = 0
	else:
		currentPatience += 1

	if(e > config.config_dic["MIN_NUM_EPOCHS"]):
	    if(currentPatience >= config.config_dic["PATIENCE"]):
	        break
	else:
		currentPatience = 0

	scheduler.step(avgValLoss)

	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	H["val_loss"].append(avgValLoss.cpu().detach().numpy())
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, config.config_dic["NUM_EPOCHS"]))
	print("Train loss: {:.6f}, Val loss: {:.4f}".format(
		avgTrainLoss, avgValLoss))


# Load weights for testing
if os.path.exists(logging_path + 'model.pth'):
    print("[INFO] Loading pre-trained weights for testing...")
    unet.load_state_dict(torch.load(logging_path + 'model.pth'))
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

		#jaccard(sigmoid(pred),y)
		if(testIndex == 0):
			num_img = np.min((x.shape[0],config.config_dic["NUM_LOG_IMAGES"]))
			sigmoid_pediction = sigmoid(pred).cpu()
			x_cpu = x.cpu()
			y_cpu = y.cpu()
			for i in range(num_img):
				#import pdb
				#pdb.set_trace()
				fig,axs = plt.subplots(1,3)
				axs[0].imshow(x_cpu[i].permute(1,2,0))
				axs[1].imshow(y_cpu[i].permute(1,2,0))
				axs[2].imshow(sigmoid_pediction[i].permute(1,2,0))
				for a in axs:
					a.set_axis_off()
				plt.tight_layout()
				wandb.log({f"testImage {i}": wandb.Image(plt)})
				plt.close()

avgTestLoss = totalTestLoss / testSteps
wandb.log({"test/avgTestLoss": avgTestLoss})
print("Train loss: {:.6f}, Val loss: {:.4f}, Test loss: {:.4f}".format(avgTrainLoss, avgValLoss, avgTestLoss))
miou = jaccard.compute()
wandb.log({"test/miou": miou})

# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

# wandb.sync()
wandb.finish()
