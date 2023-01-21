from config.path import PATH
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

from os.path import exists, join
import json
from models import HierarchicalClassifier
from data import get_hierarchical_plantnet, get_plantnet, get_genera_plantnet
#from utils.hierarchical_epoch import train_epoch, test_epoch
from utils.hierarchical_loss import HierarchicalLossNetwork


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


#LR E-3 + CosineAnnealing + baseline model + Augmentation
def baseline(exp_dir, MODEL, device):
    #experiment environment
    print(f"Using {device} device")
    set_seed(random_seed=614, use_gpu=True, dev=True) #set a random-seed for reproducible experiment.
    weight_path = f"{exp_dir}/checkpoints/checkpoint.pt"
    log_dir = f"{exp_dir}/logs/"    

    #data
    data_loaders, class_to_name = get_plantnet(transforms=transforms) #get PlantNet-300K dataset by default options.

    #model, optimizer, scheduler, earlystopping
    model = MODEL(num_classes=len(class_to_name), loss_fn=nn.CrossEntropyLoss()).to(device) #get your model
    optimizer = AdamW(model.parameters(), lr=1e-3) #get your optimizer
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1E-6) #get your lr scheduler
    early_stopping = EarlyStopping(patience=10, path=weight_path)

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

    optimizer = model.load(weight_path, optimizer) # load Its the best checkpoint.
    #test process
    test_epoch(model, data_loaders["test"], metrics["test"])
    metrics["test"].to_csv(log_dir)

#LR E-3 + CosineAnnealing + baseline model + Augmentation
def genera(exp_dir, MODEL, device):
    #experiment environment
    print(f"Using {device} device")
    set_seed(random_seed=614, use_gpu=True, dev=True) #set a random-seed for reproducible experiment.
    weight_path = f"{exp_dir}/checkpoints/checkpoint.pt"
    log_dir = f"{exp_dir}/logs/"    

    #data
    data_loaders, class_to_name = get_genera_plantnet(transforms=transforms) #get PlantNet-300K dataset by default options.
    
    #model, optimizer, scheduler, earlystopping
    model = MODEL(num_classes=len(class_to_name), loss_fn=nn.CrossEntropyLoss()).to(device) #get your model
    optimizer = AdamW(model.parameters(), lr=1e-3) #get your optimizer
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1E-6) #get your lr scheduler
    early_stopping = EarlyStopping(patience=10, path=weight_path)

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

    optimizer = model.load(weight_path, optimizer) # load Its the best checkpoint.
    #test process
    test_epoch(model, data_loaders["test"], metrics["test"])
    metrics["test"].to_csv(log_dir)


#LR E-3 + CosineAnnealing + Deep Hierarchical classifier + Augmentation
def genera_species_hierarchical_classifier(exp_dir, MODEL, device, baseline_weight_path, method=None):
    #experiment environment
    print(f"Using {device} device")
    set_seed(random_seed=614, use_gpu=True, dev=True) #set a random-seed for reproducible experiment.

    weight_path = f"{exp_dir}/{method}/checkpoints/checkpoint.pt"
    log_dir = f"{exp_dir}/{method}/logs/"   

    #data
    data_loaders, class_to_name = get_hierarchical_plantnet(transforms=transforms) #get PlantNet-300K dataset by default options.

    with open(join("data/plantnet", "fine_to_coarse.json"), 'r') as st_json:
        fine_to_coarse = json.load(st_json)
    l2_num_classes = len(set(fine_to_coarse.keys()))
    l1_num_classes = len(set(fine_to_coarse.values()))
    num_classes = [l1_num_classes, l2_num_classes]   
    
    
    if method == "scratch":
        baseline = MODEL(num_classes=len(class_to_name), loss_fn=nn.CrossEntropyLoss()) #get your model
        model = HierarchicalClassifier(loss_fn=HierarchicalLossNetwork(fine_to_coarse, device), baseline=baseline, num_classes=num_classes).to(device)
    elif method == "finetune":
        baseline = MODEL(num_classes=len(class_to_name), loss_fn=nn.CrossEntropyLoss()) #get your model
        baseline.load(baseline_weight_path) # load Its the best checkpoint.
        model = HierarchicalClassifier(loss_fn=HierarchicalLossNetwork(fine_to_coarse, device), baseline=baseline, num_classes=num_classes).to(device)
    elif method == "fixed_extractor":
        baseline = MODEL(num_classes=len(class_to_name), loss_fn=nn.CrossEntropyLoss()) #get your model
        baseline.load(baseline_weight_path) # load Its the best checkpoint.
        model = HierarchicalClassifier(loss_fn=HierarchicalLossNetwork(fine_to_coarse, device), baseline=baseline, num_classes=num_classes).to(device)
        for name, param in model.named_parameters():
            if 'baseline' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True    
    else:
        raise ValueError(f"Wrong training method {method}")
    
    optimizer = AdamW(model.parameters(), lr=1e-3) #get your optimizer
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1E-6) #get your lr scheduler
    early_stopping = EarlyStopping(patience=10, path=weight_path)

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

    optimizer = model.load(weight_path, optimizer) # load Its the best checkpoint.
    #test process
    test_epoch(model, data_loaders["test"], metrics["test"])
    metrics["test"].to_csv(log_dir)


def test(exp_dir, MODEL, device):
    #experiment environment
    print(f"Using {device} device")
    set_seed(random_seed=614, use_gpu=True, dev=True) #set a random-seed for reproducible experiment.
    weight_path = f"{exp_dir}/checkpoints/checkpoint.pt"
    log_dir = f"{exp_dir}/logs/"    
    
    #data
    data_loaders, class_to_name = get_plantnet(transforms=transforms) #get PlantNet-300K dataset by default options.    
    metric = Metric_tracker("test", class_to_name, log_dir)
    
    #model, optimizer
    model = MODEL(num_classes=len(class_to_name), loss_fn=nn.CrossEntropyLoss()).to(device) #get your model
    optimizer = AdamW(model.parameters(), lr=1e-3) #get your optimizer    
    
    optimizer = model.load(weight_path, optimizer) # load Its the best checkpoint.
    
    #test process
    test_epoch(model, data_loaders["test"], metric)
    metric.to_csv(log_dir)