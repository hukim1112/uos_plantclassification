import os
import json
from random import shuffle
import cv2
from torch.utils.data import Dataset
from os.path import join, isdir, isfile
import pickle


class PlantNet300K(Dataset):
    def __init__(self, root, split, shuffle=False, transform=None):
        self.root = root
        self.split = split
        self.shuffle = shuffle
        self.transform = transform
        self.label_to_class, self.class_to_name = self.labels()
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

        with open(join(self.root, "plantnet300K_species_id_2_name.json"), 'r') as file:
            label_to_name = json.load(file) #label => name
        
        for label, name in zip(label_to_name.keys(), label_to_name.values()):
            class_to_name[len(label_to_class)] = name
            label_to_class[label] = len(label_to_class)
        return label_to_class, class_to_name              

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

class PlantNetTarget148(PlantNet300K): #redefine the class for specific targets(148 categories of plants)
    def __init__(self, root, split, shuffle=False, transform=None):
        self.root = root
        self.split = split
        self.shuffle = shuffle
        self.transform = transform
        with open('/home/files/experiments/plantnet_embeddings_v3/label148_list.txt', 'rb') as f:
            data_list = pickle.load(f)
        self.data_list=data_list
        self.label_to_class, self.class_to_name = self.labels()
        self.filelist = self.get_filelist()
        

    def get_filelist(self): #only if in new categories(total 148), append filelist
        dir_path = join(self.root, "images", self.split)
        sub_paths = [join(dir_path, sub) for sub in os.listdir(dir_path) if (isdir(join(dir_path, sub)) and (sub in self.data_list))]

        filelist = []
        for sub_path in sub_paths:
            files = [ join(sub_path, _file)  for _file in os.listdir(sub_path) if isfile(join(sub_path, _file))]
            filelist += files
        if self.shuffle == True:
            shuffle(filelist)
        return filelist

    def labels(self): #change labels to new target classes
        label_to_class = {} # label => class
        class_to_name = {} # class => name

        with open(join(self.root, "plantnet300K_species_id_2_name.json"), 'r') as file:
            label_to_name = json.load(file) #label => name
        
        for label, name in zip(label_to_name.keys(), label_to_name.values()):
            if label not in self.data_list: 
                continue
            class_to_name[len(label_to_class)] = name
            label_to_class[label] = len(label_to_class)
        return label_to_class, class_to_name


