import torch
import json, os
from os.path import join, isdir, isfile
from os import listdir
import cv2
import numpy as np
import pickle

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
