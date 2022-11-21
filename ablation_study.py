from config.path import PATH
from os.path import join
import os
import json
import argparse
import torch
from torch import nn

from data.PlantNet import HierarchicalPlantNet, PlantNet
from experiment.plantnet import transforms
from torch.utils.data import DataLoader
from utils.epoch import train_epoch, test_epoch
from utils.metrics import Metric_tracker
from torch.optim import AdamW
from utils.earlystopping import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter


#Model import
from models import EfficientB4, HierarchicalClassifier
from test_models.ablation_models import TWO_HEAD, TWO_HEAD_CONCAT, MDHC
#Loss import
from test_models.ablation_loss import HierarchicalLossNetwork, MHLN, NO_HC_DEPENDENCY

with open(join(PATH["PLANTNET-300K"], "plantnet300K_species_id_2_name.json"), 'r') as file:
    label_to_species = json.load(file) #label => name
with open(join("data/plantnet", "species_to_fine.json"), 'r') as file:
    species_to_fine = json.load(file) #label => name
with open(join("data/plantnet", "fine_to_coarse.json"), 'r') as file:
    fine_to_coarse = json.load(file) #label => name
with open(join("data/plantnet", "genera_to_coarse.json"), 'r') as file:
    genera_to_coarse = json.load(file) #label => name
with open(join("data/plantnet", "genera_to_species.json"), 'r') as file:
    genera_to_species = json.load(file) #label => name

def metric_to_acc(metric):
    return metric.result()["topk_acc"][1].item()

def metric_to_bal_acc(metric):
    return torch.mean(metric.result()["recalls"] ).item()


def test_ablation_model(MODE, MODEL, LOSS, hierarchy, finetune, log_dir, metric_func, device):
    os.makedirs(log_dir, exist_ok=True)
    weight_path = f"{log_dir}/checkpoint.pt"
        
    # load dataset 
    data_loaders = {}
    metrics = {}

    if hierarchy:
        print(f"hierarchy is {hierarchy}")
        for split in ['train', 'val', 'test']:
            dataset = HierarchicalPlantNet(PATH["PLANTNET-300K"], split=split, fine_to_coarse=fine_to_coarse ,shuffle=(split=='train'), transform=transforms[split])
            data_loaders[split] = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
            metrics[split] = Metric_tracker(split, dataset.class_to_name, log_dir=log_dir, hierarchical=True, fine_to_coarse=fine_to_coarse)     

        baseline_weight_path = "/home/files/experiments/plantnet/baseline/EfficientB4/genera/checkpoints/checkpoint.pt"
        baseline = EfficientB4(num_classes=303, loss_fn=nn.CrossEntropyLoss()) #get your model
        baseline.load(baseline_weight_path) # load Its the best checkpoint.
        
        if MODEL == "two_head":
            net = TWO_HEAD
        elif MODEL == "two_head_concat":
            net = TWO_HEAD_CONCAT
        elif MODEL == "DHC":
            net = HierarchicalClassifier
        elif MODEL == "MDHC":
            net = MDHC
        else:
            raise ValueError(f"Wrong model name {MODEL}")

        if LOSS == "no_dependent_loss":
            loss = NO_HC_DEPENDENCY
        elif LOSS == "DLN":
            loss = HierarchicalLossNetwork
        elif LOSS == "MHLN":
            loss = MHLN
        else:
            raise ValueError(f"Wrong loss model name {LOSS}")
        
        model = net(loss(fine_to_coarse, beta="scheduler", device=device), baseline=baseline, num_classes=[len(genera_to_coarse), len(species_to_fine)]).to(device)
        print(f"finetune is {finetune}")
        if not finetune:
            for name, param in model.named_parameters():
                if 'baseline' in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True            
   
    else: #it means flat classification
        print(f"hierarchy is {hierarchy}")
        for split in ['train', 'val', 'test']:
            dataset = PlantNet(PATH["PLANTNET-300K"], split=split, shuffle=(split=='train'), transform=transforms[split])
            if split == "train":
                class_to_name = dataset.class_to_name
            data_loaders[split] = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
            metrics[split] = Metric_tracker(split, class_to_name, log_dir=log_dir, hierarchical=False, fine_to_coarse=fine_to_coarse)
        model = EfficientB4(num_classes=len(class_to_name), loss_fn=nn.CrossEntropyLoss()).to(device) #get your model
        
        print(f"finetune is {finetune}")
        if not finetune:
            for name, param in model.named_parameters():
                if 'fc' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False           
    
    if MODE == "train":    
        optimizer = AdamW(model.parameters(), lr=1e-3) #get your optimizer
        lr_scheduler = None #CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1E-6) #get your lr scheduler
        early_stopping = EarlyStopping(patience=10, path=weight_path)
        model.train()
        metrics["train"].reset()
        writer = SummaryWriter(log_dir=log_dir)
    
        # #training process
        epochs = 100
        for epoch in range(1,epochs+1):
            train_epoch(model, optimizer, data_loaders["train"], data_loaders["val"], metrics, epoch, lr_scheduler=lr_scheduler) 
            metrics["train"].to_writer(writer, epoch) #tensorboard에 기록
            metrics["val"].to_writer(writer, epoch) #tensorboard에 기록

            acc = metric_func(metrics["val"])
            early_stopping(acc, epoch, model, optimizer) #early stopper monitors validation loss and save your model. It stops training process with Its stop condition.
            if early_stopping.early_stop:
                print("Early stopping")
                break
        optimizer = model.load(weight_path, optimizer) # load Its the best checkpoint.
        #test process
        test_epoch(model, data_loaders["test"], metrics["test"])
        metrics["test"].to_csv(log_dir)

    elif MODE == "test":
        optimizer = model.load(weight_path, optimizer) # load Its the best checkpoint.
        #test process
        test_epoch(model, data_loaders["test"], metrics["test"])
        metrics["test"].to_csv(log_dir)        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This code is written for ablation study of HDC on plantnet-300k')
    # No dash argument is a positional argument and it must be input.
    # Another is a optional argument(--foo or -foo) and can be optionally input as the name.
    
    EXPEMENT_LIST = ["flat", "two_head", "two_head_concat", "DHC", "MDHC", "MDHC_focal"]
    
    parser.add_argument('experiment', type=int,
                        metavar='experiment',
                        choices=list(range(len(EXPEMENT_LIST))),
                        help='What is the exp name?')     
    parser.add_argument('mode', type=str,
                        metavar='mode',
                        choices=["train", "test"],
                        help='train or test?')    
    parser.add_argument('device', type=int,
                        metavar='cuda gpu number',
                        choices=[0,1,2],
                        help='What is the number of cuda?')

    args = parser.parse_args()
    EXP_NAME = EXPEMENT_LIST[args.experiment]
    DEVICE = f"cuda:{args.device}"
    MODE = args.mode
    #input (MODEL, LOSS, hierarchy, finetune, log_dir, metric_func, device)

    test_name = "ablation_final"
    log_dir = f"/home/files/experiments/ablation_test/{test_name}/{EXP_NAME}/"
    
    if os.path.isdir(log_dir):
        raise ValueError(f"Already experiment directory exists {log_dir}. delete it first.")
    
    answer = input(f"ARE YOU SURE to run {EXP_NAME} on cuda {DEVICE}? y, n : ")
    if answer == "y":
        pass
    else:
        raise ValueError("PROGRAM EXIT.")
    
    if EXP_NAME == "flat":
        test_ablation_model(MODE, MODEL=None, LOSS=None, hierarchy=False, finetune=True, log_dir=log_dir, metric_func=metric_to_acc, device=DEVICE)
    else:
        hierarchy = True
        if EXP_NAME == "two_head":
            MODEL = "two_head"
            LOSS = "no_dependent_loss"
        elif EXP_NAME == "two_head_concat":
            MODEL = "two_head_concat"
            LOSS = "no_dependent_loss"
        elif EXP_NAME == "DHC":
            MODEL = "DHC"
            LOSS = "DLN"
        elif EXP_NAME == "MDHC":
            MODEL = "MDHC"
            LOSS = "DLN"
        elif EXP_NAME == "MDHC_focal":
            MODEL = "MDHC"
            LOSS = "MHLN"
        else:
            raise ValueError(f"Wrong EXP_NAME : {EXP_NAME}. PROGRAM EXIT.")
        test_ablation_model(MODE, MODEL=MODEL, LOSS=LOSS, hierarchy=hierarchy, finetune=True, log_dir=log_dir, metric_func=metric_to_acc, device=DEVICE)


