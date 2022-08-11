import torch
import json, os
from os.path import join, isdir, isfile
from os import listdir
import cv2
import numpy as np

class Embedder():
    def __init__(self, root, split, transform, device):
        self.root = root
        self.split = split
        self.transform = transform
        self.device = device
        self.label_to_class, self.class_to_name = self.labels()
        self.num_classes = len(self.label_to_class)

    def get_model(self, model):
        self.model = model

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
        top_1 = []
        for file in files:
            image, class_id = self.get_img(file)
            score = self.model(image.unsqueeze(0).to(self.device))
            top_1.append(torch.argmax(score, axis=-1)[0].item())
        return top_1, top_1 == self.label_to_class[label]
    
    def extract_embeddings(self, label):
        self.model.eval()
        files = self.get_class_files(label)
        embeddings = []
        top_1 = []
        for file in files:
            image, class_id = self.get_img(file)
            with torch.no_grad():
                embedding = self.model.embedding(self.model.patch_embedding(image.unsqueeze(0).to(self.device)))
                score = self.model.fc(embedding)
            embeddings.append(embedding[0])
            top_1.append(torch.argmax(score[0]).item())
        embeddings = torch.stack(embeddings)
        top_1 = torch.tensor(top_1)
        return embeddings, top_1, top_1 == class_id

    def save_embeddings(self, path, label):
        embeddings, top_1, correctness = self.extract_embeddings(label)
        os.makedirs(join(path, "embeddings"), exist_ok=True)
        os.makedirs(join(path, "correctness"), exist_ok=True)
        np.save(join(path, "embeddings", label+".npy"), embeddings.cpu().numpy())
        np.save(join(path, "correctness", label+".npy"), correctness.cpu().numpy())