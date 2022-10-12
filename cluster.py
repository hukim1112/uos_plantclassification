from config.path import PATH
from utils.reproducibility import set_seed
from models import EfficientB4, VGG19, ResNet101, WideResNet101_2
from embeddings.extract_embeddings import make_distance_matrix, categorical_clustering

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

#experiment environment
from data import MiniPlantNet
transform = A.Compose([
        A.LongestMaxSize(max_size=500),
        A.PadIfNeeded(min_height=int(380),
        min_width=int(380),
        position='top_left',
        border_mode=cv2.BORDER_CONSTANT),
        A.CenterCrop(380,380,p=1.0),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()])

exp_path= "/home/files/experiments/mini_plantnet/baseline/EfficientB4/exp_set2" # pretrained model path
dataset = MiniPlantNet(root=PATH["PLANTNET-300K"], split="train", shuffle=False, transform=transform) #dataset to generate embeddings
Model = EfficientB4 #model type
device = "cuda:0"

print(f"Using {device} device")
set_seed(random_seed=614, use_gpu=True, dev=True) #set a random-seed for reproducible experiment.



#make distance matrix for each category
#make_distance_matrix(exp_path, dataset, Model, device)

#categorical clustering with a radius
for rad in [25, 30, 35, 40]:
        categorical_clustering(exp_path, dataset, radius=rad, initial_center_id=124)