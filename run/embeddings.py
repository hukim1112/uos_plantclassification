import numpy as np
import torch
from torch import nn
from data import MiniPlantNet
from models import EfficientB4
from torchvision import transforms
from config.path import PATH
from embeddings.extract_embeddings import Embedder
from embeddings.set_distance import SetDist
from tqdm import tqdm
from os.path import join
import pickle

def extract_embeddings(exp_path, transforms, device):
    for split in ["train", "val", "test"]:
        dataset =  MiniPlantNet(root=PATH["PLANTNET-300K"], split=split, shuffle=False, transform=transforms) #get your dataset
        model = EfficientB4(num_classes=len(dataset.label_to_class), loss_fn=nn.CrossEntropyLoss()) #get your model
        model.load(f"{exp_path}/checkpoints/checkpoint.pt") # load Its the best checkpoint.
        emb = Embedder(dataset, model, transforms, device)
        labels = list(emb.label_to_class.keys())
        for label in tqdm(labels):
            emb.save_embeddings(join(exp_path, "embeddings", split), label)

def calculate_dist_matrix(exp_path, transforms, device):
    dataset =  MiniPlantNet(root=PATH["PLANTNET-300K"], split="train", shuffle=False, transform=None) #get your dataset
    model = None
    emb = Embedder(dataset, model, transforms, device)
    labels = list(emb.label_to_class.keys())
    num_label = len(labels)
    dist_matrix = np.zeros(shape=(num_label, num_label))-1
    dist = SetDist()
    stats = {}
    for label in labels:
        print(f"calcuate stats : {label}")
        embeddings, top_1_prob, correctness, file_paths = emb.load_embeddings(join(exp_path, "embeddings", "train"), label)
        stats[label] = dist.multi_variate_gaussian(torch.tensor(embeddings))

    for i in range(dist_matrix.shape[0]):
        print(i)
        for j in range(dist_matrix.shape[1]):
            tmp = None
            label1 = labels[i]
            label2 = labels[j]
            if dist_matrix[i,j] != -1:
                tmp = dist_matrix[i,j]
            if dist_matrix[j,i] != -1:
                tmp = dist_matrix[j,i]

            if tmp is not None:
                dist_matrix[i,j] = tmp
            else:
                dist_matrix[i,j] = dist.bhattacharyya_from_stats(stats[label1], stats[label2])
        np.save(join(exp_path, f"dist_matrix.npy"), dist_matrix)

    with open(join(exp_path, f"label_{num_label}_list.txt"), "wb") as fp:
        pickle.dump(labels, fp)