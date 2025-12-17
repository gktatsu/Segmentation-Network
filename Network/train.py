# USAGE
# python train.py
# import the necessary packages
from pyimagesearch.dataset import SegmentationDataset
from pyimagesearch.model import UNet
from pyimagesearch import config
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms as T
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
from datetime import datetime
import uuid
import argparse
import wandb
from torchmetrics import JaccardIndex
from torch.optim.lr_scheduler import ReduceLROnPlateau


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the segmentation network with optional overrides."
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help=(
            "Override the base dataset directory (expects train/val/test "
            "subfolders)."
        ),
    )
    parser.add_argument(
        "--base-output",
        type=str,
        help="Override the directory used for checkpoints and logs.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Custom WandB run name (falls back to timestamp+uuid).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size defined in config.py.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Override DataLoader worker count defined in config.py.",
    )
    parser.add_argument(
        "--online-rotation-max-degrees",
        type=float,
        help="Enable online rotation augmentation with the given +/- degrees.",
    )
    parser.add_argument(
        "--online-augmentations-per-image",
        type=int,
        help=(
            "Number of rotated copies per original image when online "
            "rotation is enabled."
        ),
    )
    return parser.parse_args()


args = parse_args()
config.apply_overrides(
    DATASET_PATH=args.dataset_path,
    BASE_OUTPUT=args.base_output,
    BATCH_SIZE=args.batch_size,
    NUM_WORKERS=args.num_workers,
    ONLINE_ROTATION_MAX_DEGREES=args.online_rotation_max_degrees,
    ONLINE_AUGMENTATIONS_PER_IMAGE=args.online_augmentations_per_image,
)

# WandB API key must be set via environment variable
wandb_api_key = os.environ.get('WANDB_API_KEY')
if not wandb_api_key:
    raise EnvironmentError(
        "WANDB_API_KEY environment variable is required. "
        "Set it with: export WANDB_API_KEY='your-api-key'"
    )
wandb.login(key=wandb_api_key)

timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
unique_suffix = f"p{os.getpid()}_{uuid.uuid4().hex[:8]}"
auto_run_name = f"training_run_{timestamp_str}_{unique_suffix}"
run_name_candidates = [
    args.run_name,
    os.environ.get("RUN_NAME"),
    os.environ.get("JOB_NAME"),
]
run_name = next((name for name in run_name_candidates if name), auto_run_name)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Segmentation Network",
    name=run_name,
    # track hyperparameters and run metadata
    config=config.config_dic
)

wandb.define_metric("epoch")
wandb.define_metric("epoch/*", step_metric="epoch")


def log_metrics(data, *, epoch=None, commit=True):
    payload = dict(data)
    if epoch is not None:
        payload.update({f"epoch/{key}": value for key, value in data.items()})
        payload["epoch"] = epoch
    wandb.log(payload, commit=commit)


effective_run_name = str(wandb.run.name)
logging_path = os.path.join(
    config.config_dic["BASE_OUTPUT"], effective_run_name
)
os.makedirs(logging_path, exist_ok=True)

trainImages = os.path.join(config.config_dic["DATASET_PATH"], "train_images")
trainMasks = os.path.join(config.config_dic["DATASET_PATH"], "train_masks")
valImages = os.path.join(
    config.config_dic["DATASET_PATH"], "validation_images")
valMasks = os.path.join(config.config_dic["DATASET_PATH"], "validation_masks")
testImages = os.path.join(config.config_dic["DATASET_PATH"], "test_images")
testMasks = os.path.join(config.config_dic["DATASET_PATH"], "test_masks")

# define transformations
base_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize(
        (
            config.config_dic["INPUT_IMAGE_HEIGHT"],
            config.config_dic["INPUT_IMAGE_WIDTH"],
        )
    ),
    T.ToTensor(),
])

# create the train and test datasets
trainDS = SegmentationDataset(
    imagePaths=trainImages,
    maskPaths=trainMasks,
    transforms=base_transforms,
    rotation_degrees=config.config_dic["ONLINE_ROTATION_MAX_DEGREES"],
    augmentations_per_image=config.config_dic[
        "ONLINE_AUGMENTATIONS_PER_IMAGE"
    ],
)
valDS = SegmentationDataset(imagePaths=valImages, maskPaths=valMasks,
                            transforms=base_transforms)
testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
                             transforms=base_transforms)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(valDS)} examples in the validation set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")

# create the training and test data loaders
trainLoader = DataLoader(
    trainDS,
    shuffle=True,
    batch_size=config.config_dic["BATCH_SIZE"],
    pin_memory=config.config_dic["PIN_MEMORY"],
    num_workers=config.config_dic["NUM_WORKERS"],
)
valLoader = DataLoader(
    valDS,
    shuffle=False,
    batch_size=config.config_dic["BATCH_SIZE"],
    pin_memory=config.config_dic["PIN_MEMORY"],
    num_workers=config.config_dic["NUM_WORKERS"],
)
testLoader = DataLoader(
    testDS,
    shuffle=False,
    batch_size=config.config_dic["BATCH_SIZE"],
    pin_memory=config.config_dic["PIN_MEMORY"],
    num_workers=config.config_dic["NUM_WORKERS"],
)

# use CrossEntropyLoss for multiclass segmentation (targets are class
# indices HxW)
jaccard = JaccardIndex(
    task='multiclass',
    num_classes=config.config_dic["NUM_CLASSES"],
).to(config.config_dic['DEVICE'])

# initialize our UNet model
unet = UNet(nbClasses=config.config_dic["NUM_CLASSES"]).to(
    config.config_dic["DEVICE"])

# initialize loss function and optimizer
lossFunc = CrossEntropyLoss()
opt = Adam(unet.parameters(), lr=config.config_dic["INIT_LR"])
# calculate steps per epoch for training and test set
# len(trainLoader)
trainSteps = np.max([len(trainDS) // config.config_dic["BATCH_SIZE"], 1])
valSteps = np.max(
    [len(valDS) // config.config_dic["BATCH_SIZE"], 1])  # 25 // 64
testSteps = np.max([len(testDS) // config.config_dic["BATCH_SIZE"], 1])

# initialize a dictionary to store training history
H = {"train_loss": [], "val_loss": [], "test_loss": []}

# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
currentPatience = 0
bestValLoss = np.inf

scheduler = ReduceLROnPlateau(
    opt,
    mode='min',
    patience=config.config_dic["PATIENCE"],
    factor=config.config_dic["SCHEDULER_FACTOR"],
    verbose=True,
)

for e in tqdm(range(config.config_dic["NUM_EPOCHS"])):
    # set the model in training mode
    unet.train()
    # initialize the total training and validation loss
    totalTrainLoss = 0.0
    totalTestLoss = 0.0
    totalValLoss = 0.0

    # loop over the training set
    for (i, (x, y)) in enumerate(trainLoader):
        # send the input to the device
        (x, y) = (x.to(config.config_dic["DEVICE"]), y.to(
            config.config_dic["DEVICE"]))
        # perform a forward pass and calculate the training loss
        pred = unet(x)  # shape: (N, C, H, W), raw logits
        # import pdb
        # pdb.set_trace()
        # y is expected to be LongTensor of shape (N, H, W) with class indices
        loss = lossFunc(pred, y)
        # first, zero out any previously accumulated gradients, then
        # perform backpropagation, and then update model parameters
        opt.zero_grad()
        loss.backward()
        opt.step()
        # add the loss to the total training loss so far
        loss_value = loss.detach().item()
        totalTrainLoss += loss_value

        log_metrics({"train/loss": loss_value})

    # switch off autograd for validation
    with torch.no_grad():
        # set the model in evaluation mode
        unet.eval()
        # loop over the validation set
        for valIndex, (x, y) in enumerate(valLoader):
            # send the input to the device
            (x, y) = (x.to(config.config_dic["DEVICE"]),
                      y.to(config.config_dic["DEVICE"]))
            # make the predictions and calculate the validation loss
            pred = unet(x)
            val_loss = lossFunc(pred, y).detach().item()
            totalValLoss += val_loss
            # for multiclass Jaccard, pass predicted class indices and
            # target labels
            pred_labels = torch.argmax(pred, dim=1)
            jaccard(pred_labels, y)
            if valIndex == 0:
                num_img = np.min(
                    (x.shape[0], config.config_dic["NUM_LOG_IMAGES"])
                )
                x_cpu = x.cpu()
                y_cpu = y.cpu()
                pred_cpu = pred_labels.cpu()
                for i in range(num_img):
                    fig, axs = plt.subplots(1, 3)
                    axs[0].imshow(x_cpu[i].permute(1, 2, 0))
                    axs[1].imshow(
                        y_cpu[i],
                        cmap="tab10",
                        vmin=0,
                        vmax=config.config_dic["NUM_CLASSES"] - 1,
                    )
                    axs[2].imshow(
                        pred_cpu[i],
                        cmap="tab10",
                        vmin=0,
                        vmax=config.config_dic["NUM_CLASSES"] - 1,
                    )
                    for a in axs:
                        a.set_axis_off()
                    plt.tight_layout()
                    log_metrics(
                        {f"validationImage {i}": wandb.Image(plt)},
                        epoch=e + 1,
                    )
                    plt.close()

    # calculate the average training and validation loss
    # print("valSteps: ", valSteps)
    # print("totalTrainLoss:", totalTrainLoss)
    # print("totalValLoss", totalValLoss)
    avgTrainLoss = totalTrainLoss / trainSteps
    avgValLoss = totalValLoss / valSteps

    miou = jaccard.compute()
    miou_value = (
        miou.detach().item() if isinstance(miou, torch.Tensor) else miou
    )

    log_metrics(
        {
            "train/loss": avgTrainLoss,
            "train/avgTrainLoss": avgTrainLoss,
            "val/avgValLoss": avgValLoss,
            "val/miou": miou_value,
        },
        epoch=e + 1,
    )
    jaccard.reset()

    min_delta = config.config_dic.get("MIN_DELTA", 0.0)
    # consider an improvement only if avgValLoss improves by at least min_delta
    if avgValLoss < bestValLoss - min_delta:
        bestValLoss = avgValLoss
        # import pdb
        # pdb.set_trace()
        torch.save(unet.state_dict(), os.path.join(logging_path, "model.pth"))
        # torch.save(unet, config.MODEL_PATH)
        currentPatience = 0
    else:
        currentPatience += 1

    if e > config.config_dic["MIN_NUM_EPOCHS"]:
        if currentPatience >= config.config_dic["PATIENCE"]:
            break
    else:
        currentPatience = 0

    scheduler.step(avgValLoss)

    # update our training history
    H["train_loss"].append(avgTrainLoss)
    H["val_loss"].append(avgValLoss)
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(e + 1, config.config_dic["NUM_EPOCHS"]))
    print("Train loss: {:.6f}, Val loss: {:.4f}".format(
        avgTrainLoss, avgValLoss))


# Load weights for testing
if os.path.exists(os.path.join(logging_path, 'model.pth')):
    print("[INFO] Loading pre-trained weights for testing...")
    unet.load_state_dict(torch.load(os.path.join(logging_path, 'model.pth')))
else:
    print("[WARNING] Weights not found.")

jaccard.reset()

with torch.no_grad():
    # set the model in evaluation mode
    unet.eval()
    # loop over the test set
    for testIndex, (x, y) in enumerate(testLoader):
        # send the input to the device
        (x, y) = (x.to(config.config_dic["DEVICE"]), y.to(
            config.config_dic["DEVICE"]))
        # make the predictions and calculate the validation loss
        pred = unet(x)
        totalTestLoss += lossFunc(pred, y).detach().item()
        pred_labels = torch.argmax(pred, dim=1)
        jaccard(pred_labels, y)
        if testIndex == 0:
            num_img = np.min((x.shape[0], config.config_dic["NUM_LOG_IMAGES"]))
            x_cpu = x.cpu()
            y_cpu = y.cpu()
            pred_cpu = pred_labels.cpu()
            for i in range(num_img):
                fig, axs = plt.subplots(1, 3)
                axs[0].imshow(x_cpu[i].permute(1, 2, 0))
                axs[1].imshow(
                    y_cpu[i],
                    cmap="tab10",
                    vmin=0,
                    vmax=config.config_dic["NUM_CLASSES"] - 1,
                )
                axs[2].imshow(
                    pred_cpu[i],
                    cmap="tab10",
                    vmin=0,
                    vmax=config.config_dic["NUM_CLASSES"] - 1,
                )
                for a in axs:
                    a.set_axis_off()
                plt.tight_layout()
                log_metrics({f"testImage {i}": wandb.Image(plt)})
                plt.close()

avgTestLoss = totalTestLoss / testSteps
log_metrics({"test/avgTestLoss": avgTestLoss})
print("Train loss: {:.6f}, Val loss: {:.4f}, Test loss: {:.4f}".format(
    avgTrainLoss, avgValLoss, avgTestLoss))
miou = jaccard.compute()
miou_test_value = (
    miou.detach().item() if isinstance(miou, torch.Tensor) else miou
)
log_metrics({"test/miou": miou_test_value})

# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
    endTime - startTime))

# wandb.sync()
wandb.finish()
