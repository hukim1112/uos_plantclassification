from config.path import PATH
from os.path import join
import os
import json
import argparse
import torch
from torch import nn

from data.PlantNet import HierarchicalPlantNet, PlantNet
from config.path import PATH
from experiment.plantnet import transforms
from torch.utils.data import DataLoader

from models.Efficientnet import EfficientB4
from models.ImageClassifier import HierarchicalClassifier
from utils.hierarchical_loss import HierarchicalLossNetwork
from test_models import MDHC, MDHC_v2
from test_models import MHLN

from utils.epoch import train_epoch, test_epoch
from utils.metrics import Metric_tracker
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim import AdamW
from utils.earlystopping import EarlyStopping
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

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
    
#각 속에 속한 종의 숫자 계산
species_count = {genera:len(genera_to_species[genera]) for genera in list(genera_to_species.keys())}
sorted_species_count = sorted(species_count.items(), key = lambda item: item[1], reverse = True)

# 포함된 종의 숫자에 따른 속 분류
num_to_genera = {}
for genera, num in sorted_species_count:
    if num in num_to_genera.keys():
        num_to_genera[num].append(genera)
    else:
        num_to_genera[num] = [genera]

# 실험할 속 결정
test_genera = []
for k in num_to_genera.keys():
    test_genera += num_to_genera[k]
print(test_genera)

# 실험할 종 결정
species_list = []
for genera_name in test_genera:
    species_list+= genera_to_species[genera_name]

def get_new_fine_to_coarse(class_to_name, genera_list):
    fine_to_coarse = {}
    for c, n in class_to_name.items():
        fine_to_coarse[str(c)] = test_genera.index(n.split(" ")[0])
    return fine_to_coarse

def metric_to_acc(metric):
    return metric.result()["topk_acc"][1].item()

def metric_to_bal_acc(metric):
    return torch.mean(metric.result()["recalls"] ).item()


def test_modified_MDHC_v2(log_dir, finetune, metric_func, beta, device):
    os.makedirs(log_dir, exist_ok=True)
    weight_path = f"{log_dir}/checkpoint.pt"
        
    # load dataset 
    data_loaders = {}
    metrics = {}

    for split in ['train', 'val', 'test']:
        dataset = HierarchicalPlantNet(PATH["PLANTNET-300K"], split=split, fine_to_coarse=fine_to_coarse ,shuffle=(split=='train'), transform=transforms[split])
        dataset.labels(species_list)
        new_fine_to_coarse = get_new_fine_to_coarse(dataset.class_to_name, test_genera)
        dataset.fine_to_coarse = new_fine_to_coarse
        dataset.filelist = dataset.get_filelist()

        data_loaders[split] = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        metrics[split] = Metric_tracker(split, dataset.class_to_name, log_dir=log_dir, hierarchical=True, fine_to_coarse=new_fine_to_coarse)        
    
    baseline_weight_path = "/home/files/experiments/plantnet/baseline/EfficientB4/genera/checkpoints/checkpoint.pt"
    baseline = EfficientB4(num_classes=303, loss_fn=nn.CrossEntropyLoss()) #get your model
    baseline.load(baseline_weight_path) # load Its the best checkpoint.
    
    model = MDHC_v2(loss_fn=HierarchicalLossNetwork(new_fine_to_coarse, beta=beta, device=device), baseline=baseline, num_classes=[len(test_genera), len(species_list)]).to(device)

    if not finetune:
        for name, param in model.named_parameters():
            if 'baseline' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    optimizer = AdamW(model.parameters(), lr=1e-3) #get your optimizer
    lr_scheduler = None #CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1E-6) #get your lr scheduler
    early_stopping = EarlyStopping(patience=10, is_loss=False, path=weight_path)
    
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
def test_modified_MDHC_v2_with_MDLN(log_dir, finetune, metric_func, beta, device):
    os.makedirs(log_dir, exist_ok=True)
    weight_path = f"{log_dir}/checkpoint.pt"
        
    # load dataset 
    data_loaders = {}
    metrics = {}

    for split in ['train', 'val', 'test']:
        dataset = HierarchicalPlantNet(PATH["PLANTNET-300K"], split=split, fine_to_coarse=fine_to_coarse ,shuffle=(split=='train'), transform=transforms[split])
        dataset.labels(species_list)
        new_fine_to_coarse = get_new_fine_to_coarse(dataset.class_to_name, test_genera)
        dataset.fine_to_coarse = new_fine_to_coarse
        dataset.filelist = dataset.get_filelist()

        data_loaders[split] = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        metrics[split] = Metric_tracker(split, dataset.class_to_name, log_dir=log_dir, hierarchical=True, fine_to_coarse=new_fine_to_coarse)        
    
    baseline_weight_path = "/home/files/experiments/plantnet/baseline/EfficientB4/genera/checkpoints/checkpoint.pt"
    baseline = EfficientB4(num_classes=303, loss_fn=nn.CrossEntropyLoss()) #get your model
    baseline.load(baseline_weight_path) # load Its the best checkpoint.
    
    model = MDHC_v2(loss_fn=MHLN(new_fine_to_coarse, beta=beta, device=device), baseline=baseline, num_classes=[len(test_genera), len(species_list)]).to(device)

    if not finetune:
        for name, param in model.named_parameters():
            if 'baseline' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    optimizer = AdamW(model.parameters(), lr=1e-3) #get your optimizer
    lr_scheduler = None #CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1E-6) #get your lr scheduler
    early_stopping = EarlyStopping(patience=10, is_loss=False, path=weight_path)
    
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='This code is written for practice about argparse')
    # No dash argument is a positional argument and it must be input.
    # Another is a optional argument(--foo or -foo) and can be optionally input as the name
    parser.add_argument('device', type=int,
                        metavar='cuda gpu number',
                        choices=[0,1,2],
                        help='What is the number of cuda?')
    # parser.add_argument('test_name', type=str,
    #                     metavar='test_name',
    #                     help='What is the test name?')
    parser.add_argument('exp_name', type=str,
                        metavar='exp_name',
                        help='What is the exp name?')
    # parser.add_argument('finetune', type=bool,
    #                     metavar='finetune',
    #                     choices=[True,False],
    #                     help='Whether fintune or not')    
    parser.add_argument('loss_type', type=str,
                        metavar='loss_type',
                        choices=["DLN","MDLN"],
                        help='What is the loss type?')    
    
    
    args = parser.parse_args()
    device = f"cuda:{args.device}"
    test_name = "test_MDHC" #args.test_name
    exp_name = args.exp_name
    finetune = False #args.finetune
    loss_type = args.loss_type
    metric_func = metric_to_acc
    beta = "scheduler"
    print(device, test_name, exp_name, finetune, loss_type)
    
    # device = "cuda:0"
    # test_name = "modified_DHC"
    # finetune = False
    # metric_func = metric_to_acc
    
    log_dir = f"/home/files/experiments/ablation_test/{test_name}/{exp_name}/"
    if loss_type=="DLN":   
        test_modified_MDHC_v2(log_dir, finetune, metric_func, beta=beta, device=device) 
    elif loss_type=="MDLN":
        test_modified_MDHC_v2_with_MDLN(log_dir, finetune, metric_func, beta=beta, device=device)
    else:
        print("wrong loss type")