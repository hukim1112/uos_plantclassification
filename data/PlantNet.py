import os
import json
from random import shuffle
import cv2
from torch.utils.data import Dataset
from os.path import join, isdir, isfile, basename
import pickle


class PlantNet300K(Dataset):
    def __init__(self, root, split, shuffle=False, transform=None):
        self.root = root
        self.split = split
        self.shuffle = shuffle
        self.transform = transform
        self.label_to_class, self.class_to_name, self.name_to_label = self.labels()
        self.filelist = self.get_filelist()

    def get_filelist(self):
        dir_path = join(self.root, "images", self.split)
        sub_paths = [join(dir_path, sub) for sub in os.listdir(dir_path) if isdir(join(dir_path, sub))]

        filelist = []
        for sub_path in sub_paths:
            files = [ join(sub_path, _file)  for _file in os.listdir(sub_path) if isfile(join(sub_path, _file))]
            filelist += files
        if self.shuffle == True:
            shuffle(filelist)
        return filelist

    def labels(self):
        # example.
        # label : "1355868" 
        # name : "Lactuca virosa L."
        # class : 0

        label_to_class = {} # label => class
        class_to_name = {} # class => name
        name_to_label = {} # name => label

        with open(join(self.root, "plantnet300K_species_id_2_name.json"), 'r') as file:
            label_to_name = json.load(file) #label => name
        
        for label, name in zip(label_to_name.keys(), label_to_name.values()):
            class_to_name[len(label_to_class)] = name
            label_to_class[label] = len(label_to_class)
            name_to_label[name] = label
        return label_to_class, class_to_name, name_to_label

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        img_path = self.filelist[idx]
        image = cv2.imread(img_path)
        if image is None:
            print(f"Fail to load image file : {img_path}")
        image = image[:,:,::-1]

        label = img_path.split('/')[-2]
        class_id = self.label_to_class[label]
        if self.transform: #transform must be albumentation's tranform.
            image = self.transform(image=image)["image"]
        return image, class_id

class MiniPlantNet(PlantNet300K):
    def __init__(self, root, split, shuffle=False, transform=None, minimum_samples=32):
        self.root = root
        self.split = split
        self.shuffle = shuffle
        self.transform = transform
        self.minimum = minimum_samples
        self.label_to_class, self.class_to_name, self.name_to_label = self.labels()
        self.filelist = self.get_filelist()

    def get_filelist(self):
        dir_path = join(self.root, "images", self.split)
        sub_paths = [join(dir_path, sub) for sub in os.listdir(dir_path) if isdir(join(dir_path, sub)) and (sub in list(self.label_to_class.keys()))]

        filelist = []
        for sub_path in sub_paths:
            files = [ join(sub_path, _file)  for _file in os.listdir(sub_path) if isfile(join(sub_path, _file))]
            filelist += files
        if self.shuffle == True:
            shuffle(filelist)
        return filelist

    def labels(self): #change labels to new target classes
        dir_path = join(self.root, "images", self.split)
        sub_paths = [join(dir_path, sub) for sub in os.listdir(dir_path) if isdir(join(dir_path, sub))]
        target_labels = [os.path.basename(sub_path) for sub_path in sub_paths if len(os.listdir(sub_path))>=self.minimum]

        label_to_class = {} # label => class
        class_to_name = {} # class => name
        name_to_label = {} # name => label

        with open(join(self.root, "plantnet300K_species_id_2_name.json"), 'r') as file:
            label_to_name = json.load(file) #label => name
        
        for label, name in zip(label_to_name.keys(), label_to_name.values()):
            if label not in target_labels: 
                continue
            class_to_name[len(label_to_class)] = name
            label_to_class[label] = len(label_to_class)
            name_to_label[name] = label
        return label_to_class, class_to_name, name_to_label         

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        img_path = self.filelist[idx]
        image = cv2.imread(img_path)
        if image is None:
            print(f"Fail to load image file : {img_path}")
        image = image[:,:,::-1]

        label = img_path.split('/')[-2]
        class_id = self.label_to_class[label]
        if self.transform: #transform must be albumentation's tranform.
            image = self.transform(image=image)["image"]
        return image, class_id