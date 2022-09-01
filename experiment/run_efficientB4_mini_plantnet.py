from statistics import mean
from data import get_plantnet_for148
from models import EfficientB4
from utils.reproducibility import set_seed
from utils.metrics import Metric_tracker
from utils.earlystopping import EarlyStopping
from utils.epoch import train_epoch, test_epoch
import cv2
import torch
from torch.optim import AdamW
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2

default_transforms = {
    'train': A.Compose([
        A.Resize(height=380, width=380),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()]),
    'val': A.Compose([
        A.Resize(height=380, width=380),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()]),
    'test': A.Compose([
        A.Resize(height=380, width=380),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()])
}

#LR E-3 + CosineAnnealing + EfficientB4 
def exp_set1(weight_dir, log_dir):
    #experiment environment
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    set_seed(random_seed=614, use_gpu=True, dev=True) #set a random-seed for reproducible experiment.

    #data
    data_loaders, class_to_name = get_plantnet_for148(transforms=default_transforms) #get PlantNet-300K dataset by default options.
    
    
    #model, optimizer, scheduler, earlystopping
    model = EfficientB4(num_classes=len(class_to_name), loss_fn=nn.CrossEntropyLoss()).to(device) #get your model
    optimizer = AdamW(model.parameters(), lr=1e-3) #get your optimizer
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1, eta_min=0.0) #get your lr scheduler
    early_stopping = EarlyStopping(patience=30, path=weight_dir)

    #tensorboard logger and metrics.
    writer = SummaryWriter(log_dir=log_dir)
    metrics = {}
    for split in ["train", "val", "test"]:
        metrics[split] = Metric_tracker(split, class_to_name, log_dir)

    # #training process
    epochs = 100
    for epoch in range(1,epochs+1):
        train_epoch(model, optimizer, data_loaders["train"], data_loaders["val"], metrics, epoch, lr_scheduler=scheduler) 
        metrics["train"].to_writer(writer, epoch) #tensorboard에 기록
        metrics["val"].to_writer(writer, epoch) #tensorboard에 기록

        early_stopping(metrics["val"].cal_epoch_loss(), epoch, model, optimizer) #early stopper monitors validation loss and save your model. It stops training process with Its stop condition.
        if early_stopping.early_stop:
            print("Early stopping")
            break

    optimizer = model.load(weight_dir, optimizer) # load Its the best checkpoint.
    #test process
    test_epoch(model, data_loaders["test"], metrics["test"])
    metrics["test"].to_csv(log_dir)

#LR E-3 + ReducePlateau + EfficientB4+ Augmentation
def exp_set2(weight_dir, log_dir):
    #experiment environment
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    set_seed(random_seed=614, use_gpu=True, dev=True) #set a random-seed for reproducible experiment.

    #augmentation
    transforms = {
    'train': A.Compose([
            A.LongestMaxSize(max_size=500),
            A.PadIfNeeded(min_height=int(380),
            min_width=int(380),
            position='top_left',
            border_mode=cv2.BORDER_CONSTANT),
            A.RandomCrop(380,380,p=1.0),
            A.HorizontalFlip(0.5),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2()]),
    'val': A.Compose([
            A.LongestMaxSize(max_size=500),
            A.PadIfNeeded(min_height=int(380),
            min_width=int(380),
            position='top_left',
            border_mode=cv2.BORDER_CONSTANT),
            A.CenterCrop(380,380, p=1.0),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2()]),
    'test': A.Compose([
            A.LongestMaxSize(max_size=500),
            A.PadIfNeeded(min_height=int(380),
            min_width=int(380),
            position='top_left',
            border_mode=cv2.BORDER_CONSTANT),
            A.CenterCrop(380,380,p=1.0),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2()])
            }

    #data
    data_loaders, class_to_name = get_plantnet_for148(transforms=transforms) #get PlantNet-300K dataset by default options.

    #model, optimizer, scheduler, earlystopping
    model = EfficientB4(num_classes=len(class_to_name), loss_fn=nn.CrossEntropyLoss()).to(device) #get your model
    optimizer = AdamW(model.parameters(), lr=1e-3) #get your optimizer
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=30, path=weight_dir)

    #tensorboard logger and metrics.
    writer = SummaryWriter(log_dir=log_dir)
    metrics = {}
    for split in ["train", "val", "test"]:
        metrics[split] = Metric_tracker(split, class_to_name, log_dir)

    # #training process
    epochs = 100
    for epoch in range(1,epochs+1):
        train_epoch(model, optimizer, data_loaders["train"], data_loaders["val"], metrics, epoch, lr_scheduler=None)
        scheduler.step(metrics["val"].cal_epoch_loss())
        metrics["train"].to_writer(writer, epoch) #tensorboard에 기록
        metrics["val"].to_writer(writer, epoch) #tensorboard에 기록

        early_stopping(metrics["val"].cal_epoch_loss(), epoch, model, optimizer) #early stopper monitors validation loss and save your model. It stops training process with Its stop condition.
        if early_stopping.early_stop:
            print("Early stopping")
            break

    optimizer = model.load(weight_dir, optimizer) # load Its the best checkpoint.
    #test process
    test_epoch(model, data_loaders["test"], metrics["test"])
    metrics["test"].to_csv(log_dir)
