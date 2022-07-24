from data import get_plantnet
from models import EfficientB0
from utils.reproducibility import set_seed
from utils.metrics import Metric_tracker
from experiment.epoch import train_epoch

import torch
from torch.optim import AdamW
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter


def train(weight_dir, log_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    set_seed(random_seed=614, use_gpu=(device=="cuda"), dev=True) #set a random-seed for reproducible experiment.

    data_loaders, class_to_name = get_plantnet() #get PlantNet-300K dataset by default options.
    model = EfficientB0(num_classes=len(class_to_name), loss_fn=nn.CrossEntropyLoss()).to(device) #get your model
    optimizer = AdamW(model.parameters(), lr=1e-3) #get your optimizer
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1, eta_min=0.0) #get your lr scheduler
    writer = SummaryWriter(log_dir=log_dir)

    metrics = {"train" : Metric_tracker(), "val" : Metric_tracker(), "test" : Metric_tracker()}
    epochs = 10
    for epoch in range(1,epochs+1)
        train_epoch(model, optimizer, data_loaders["train"], data_loaders["val"], metrics, lr_scheduler=None) 
    metrics["train"].to_writer(writer)
    metrics["val"].to_writer(writer)