import numpy as np
import torch
from torch import nn
from data import MiniPlantNet
from models import EfficientB4
from torchvision import transforms
from config.path import PATH
from embeddings.extract_embeddings import Embedder, extract_embeddings, calculate_dist_matrix, summon_embedder, category_clustering
from embeddings.set_distance import SetDist
from embeddings.clustering import K_center_greedy

from tqdm import tqdm
from os.path import join
import pickle
import json

def func(exp_path, transforms, Model, device):
    extract_embeddings(exp_path, transforms, Model, device)
    calculate_dist_matrix(exp_path, transforms, device)
    emb, label_to_name = summon_embedder(exp_path, transforms, device)
    category_clustering(exp_path, emb.class_to_name, threshold=25, initial_center_id=124)






#experiment settings
import albumentations as A
from albumentations.pytorch import ToTensorV2
exp_path = "/home/files/experiments/mini_plantnet/efficientB4/exp2"
transforms = A.Compose([
        A.Resize(height=380, width=380),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()])

A.Compose([
            A.LongestMaxSize(max_size=500),
            A.PadIfNeeded(min_height=int(380),
            min_width=int(380),
            position='top_left',
            border_mode=cv2.BORDER_CONSTANT),
            A.CenterCrop(380,380,p=1.0),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2()])        
     
device='cuda:1'

#extract_embeddings
#extract_embeddings(exp_path, transforms, device)
#calculate_dist_matrix(exp_path, transforms, device)

#clustering
emb, label_to_name = summon_embedder(exp_path, transforms, device)
category_clustering(exp_path, emb.class_to_name, threshold=25, initial_center_id=124)