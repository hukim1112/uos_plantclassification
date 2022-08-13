import torch
import json, os
from os.path import join, isdir, isfile
from os import listdir
import cv2
import numpy as np
import pickle

class Embedder():
    def __init__(self, root, split):
        self.root = root
        self.split = split
        self.label_to_class, self.class_to_name = self.labels()
        self.num_classes = len(self.label_to_class)
        self.model = None

    def get_model(self, model, transform, device):
        self.model = model.to(device)
        self.transform = transform
        self.device = device

    def labels(self):
        # example.
        # label : "1355868" 
        # name : "Lactuca virosa L."
        # class : 0

        label_to_class = {} # label => class
        class_to_name = {} # class => name

        with open(join(self.root, "plantnet300K_species_id_2_name.json"), 'r') as file:
            label_to_name = json.load(file) #label => name
        
        for label, name in zip(label_to_name.keys(), label_to_name.values()):
            class_to_name[len(label_to_class)] = name
            label_to_class[label] = len(label_to_class)
        return label_to_class, class_to_name
    
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
            image = self.transform(image)
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
        top_1 = []
        file_paths = []
        for file in files:
            image, class_id = self.get_img(file)
            with torch.no_grad():
                embedding = self.model.embedding(self.model.patch_embedding(image.unsqueeze(0).to(self.device)))
                score = self.model.fc(embedding)
            embeddings.append(embedding[0])
            top_1.append(torch.argmax(score[0]).item())
            file_paths.append(file)
        embeddings = torch.stack(embeddings)
        top_1 = torch.tensor(top_1)
        return embeddings, top_1, top_1 == class_id, file_paths

    def save_embeddings(self, path, label):
        embeddings, top_1, correctness, file_paths = self.extract_embeddings(label)
        os.makedirs(join(path, "embeddings"), exist_ok=True)
        os.makedirs(join(path, "correctness"), exist_ok=True)
        os.makedirs(join(path, "filename"), exist_ok=True)
        np.save(join(path, "embeddings", label+".npy"), embeddings.cpu().numpy())
        np.save(join(path, "correctness", label+".npy"), correctness.cpu().numpy())
        with open(join(path, "filename", label), "wb") as fp:
            pickle.dump(file_paths, fp)

    def load_embeddings(self, path, label):
        embeddings = np.load(join(path, "embeddings", label+".npy"))
        correctness = np.load(join(path, "correctness", label+".npy"))
        with open(join(path, "filename", label), "rb") as fp:
            file_paths = pickle.load(fp)
        return embeddings, correctness, file_paths

'''
import torch
from torch import nn
from models import EfficientB4
from torchvision import transforms
from config.path import PATH
from experiment.extract_embeddings import Embedder
from tqdm import tqdm

root = PATH["PLANTNET-300K"]
transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((380, 380)),
            transforms.ToTensor()])
device = "cuda:0"
weight_dir = "/home/files/experiments/efficientB4/exp_set3/checkpoints/checkpoint.pt"


from os.path import join
for split in ["train", "val", "test"]:
    emb = Embedder(root, split, transform, device)
    labels = list(emb.label_to_class.keys())
    model = EfficientB4(num_classes=emb.num_classes, loss_fn=nn.CrossEntropyLoss()) #get your model
    model.load(weight_dir) # load Its the best checkpoint.
    emb.get_model(model.to(device))

    for label in tqdm(labels):
        emb.save_embeddings(join("/home/files/experiments/plantnet_embeddings", split), label)
'''