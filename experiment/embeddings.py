import numpy as np
import torch
from torch import nn
from data import MiniPlantNet
from models import EfficientB4
from torchvision import transforms
from config.path import PATH
from embeddings.extract_embeddings import Embedder
from embeddings.set_distance import SetDist
from embeddings.clustering import K_center_greedy

from tqdm import tqdm
from os.path import join
import pickle
import json

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
        np.save(join(exp_path, "dist_matrix.npy"), dist_matrix)

    with open(join(exp_path, "label_list.txt"), "wb") as fp:
        pickle.dump(labels, fp)

def calculate_dists_each_label_from_matrix(exp_path, transforms, device):
    dataset =  MiniPlantNet(root=PATH["PLANTNET-300K"], split="train", shuffle=False, transform=None) #get your dataset
    model = None
    emb = Embedder(dataset, model, transforms, device)
    labels = list(emb.label_to_class.keys())

    with open(join(exp_path, "label_list.txt"), "rb") as fp:
        labels = pickle.load(fp)
    dist_matrix = np.load(join(exp_path, "dist_matrix.npy"))

    # name_to_label = emb.name_to_label
    # name_to_label = dict([(value, key) for key, value in name_to_label.items()])

    dists_from_label = {}
    for i in range(dist_matrix.shape[0]):
        target_label = labels[i]
        target_dists = dist_matrix[i]
        sorted_idx = np.argsort(target_dists)
        sorted_dists = target_dists[sorted_idx]
        sorted_labels = np.array(labels)[sorted_idx]
        dists_from_label[target_label] = {"target_label" : target_label, "dists" : list(sorted_dists), "labels" : list(sorted_labels)}

    with open(join(exp_path, "dists_each_label.json"), 'w', encoding='utf-8') as f: 
        json.dump(dists_from_label, f, ensure_ascii=False, indent=2)
        
def summon_embedder(exp_path, transforms, device):
    dataset =  MiniPlantNet(root=PATH["PLANTNET-300K"], split="train", shuffle=False, transform=None) #get your dataset
    model = None
    emb = Embedder(dataset, model, transforms, device)
    labels = list(emb.label_to_class.keys())
    label_to_name = dict([(value, key) for key, value in emb.name_to_label.items()])
    return emb, label_to_name

def category_clustering(exp_path, class_to_name, threshold, initial_center_id):
    dist_matrix = np.load(join(exp_path, "dist_matrix.npy"))
    cluster = K_center_greedy(dist_matrix, threshold=threshold, initial_center_id=initial_center_id)
    clusters, name_clusters, fine_to_coarse, center_to_coarse = cluster.get_clusters(class_to_name)
    with open(join(exp_path, f"clusters_{threshold}.json"), 'w', encoding='utf-8') as f: 
        json.dump(clusters, f, ensure_ascii=False, indent=2)
    with open(join(exp_path, f"name_clusters_{threshold}.json"), 'w', encoding='utf-8') as f: 
        json.dump(name_clusters, f, ensure_ascii=False, indent=2)
    with open(join(exp_path, f"fine_to_coarse_{threshold}.json"), 'w', encoding='utf-8') as f: 
        json.dump(fine_to_coarse, f, ensure_ascii=False, indent=2)
    with open(join(exp_path, f"center_to_coarse_{threshold}.json"), 'w', encoding='utf-8') as f: 
        json.dump(center_to_coarse, f, ensure_ascii=False, indent=2)