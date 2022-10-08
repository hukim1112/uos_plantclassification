import torch
import json, os
from os.path import join, isdir, isfile
from os import listdir
import cv2
import numpy as np
import pickle
from tqdm import tqdm

from torchvision import transforms
from config.path import PATH

from .set_distance import SetDist
from .clustering import K_center_greedy
from data import MiniPlantNet

class Embedder():
    def __init__(self, dataset, model, transform, device):
        self.root = dataset.root
        self.split = dataset.split
        self.label_to_class = dataset.label_to_class
        self.class_to_name = dataset.class_to_name
        self.name_to_label = dataset.name_to_label
        self.num_classes = len(self.label_to_class)
        self.transform = transform
        self.device = device
        if model is not None:
            self.model = model.to(device)
        else:
            self.model = model
    def get_class_files(self, label):
        dir_path = join(self.root, "images", self.split, label)
        return [join(dir_path, file) for file in listdir(dir_path)]

    def get_img(self, img_path):
        image = cv2.imread(img_path)
        if image is None:
            print(f"Fail to load image file : {img_path}")
        image = image[:,:,::-1]

        label = img_path.split('/')[-2]
        class_id = self.label_to_class[label]
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, class_id

    def extract_predictions(self, label):
        self.model.eval()
        files = self.get_class_files(label)
        top_1 = [],
        file_paths = []
        for file in files:
            image, class_id = self.get_img(file)
            score = self.model(image.unsqueeze(0).to(self.device))
            top_1.append(torch.argmax(score, axis=-1)[0].item())
            file_paths.apppend(file)
        return top_1 == class_id
    
    def extract_embeddings(self, label):
        if self.model is None:
            raise ValueError("You need to call get_model() to load your model first.")
        self.model.eval()
        files = self.get_class_files(label)
        embeddings = []
        top_1_class = []
        top_1_prob = []
        file_paths = []
        for file in files:
            image, class_id = self.get_img(file)
            with torch.no_grad():
                embedding = self.model.embedding(self.model.patch_embedding(image.unsqueeze(0).to(self.device)))
                score = self.model.fc(embedding)
                probs = torch.softmax(score, dim=-1)
            embeddings.append(embedding[0])
            top_1_class.append(torch.argmax(score[0]).item())
            top_1_prob.append(torch.max(probs[0]).item())
            file_paths.append(file)
        embeddings = torch.stack(embeddings)
        top_1_class = torch.tensor(top_1_class)
        top_1_prob = torch.tensor(top_1_prob)
        correctness = top_1_class == class_id
        return embeddings, top_1_class, top_1_prob, correctness, file_paths

    def save_embeddings(self, path, label):
        embeddings, top_1_class, top_1_prob, correctness, file_paths = self.extract_embeddings(label)
        os.makedirs(join(path, "embeddings"), exist_ok=True)
        os.makedirs(join(path, "correctness"), exist_ok=True)
        os.makedirs(join(path, "filename"), exist_ok=True)
        os.makedirs(join(path, "top_1_probability"), exist_ok=True)
        np.save(join(path, "embeddings", label+".npy"), embeddings.cpu().numpy())
        np.save(join(path, "top_1_probability", label+".npy"), top_1_prob.cpu().numpy())
        np.save(join(path, "correctness", label+".npy"), correctness.cpu().numpy())
        with open(join(path, "filename", label), "wb") as fp:
            pickle.dump(file_paths, fp)

    def load_embeddings(self, path, label):
        embeddings = np.load(join(path, "embeddings", label+".npy"))
        top_1_prob = np.load(join(path, "top_1_probability", label+".npy"))
        correctness = np.load(join(path, "correctness", label+".npy"))
        with open(join(path, "filename", label), "rb") as fp:
            file_paths = pickle.load(fp)
        return embeddings, top_1_prob, correctness, file_paths

def extract_embeddings(exp_path, transforms, Model, device):
    for split in ["train", "val", "test"]:
        dataset =  MiniPlantNet(root=PATH["PLANTNET-300K"], split=split, shuffle=False, transform=transforms) #get your dataset
        model = Model(num_classes=len(dataset.label_to_class), loss_fn=nn.CrossEntropyLoss()) #get your model
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