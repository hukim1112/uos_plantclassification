import os
import json
from random import shuffle
import cv2
from torch.utils.data import Dataset
from os.path import join, isdir, isfile, basename
import pickle


class PlantNet(Dataset):
    def __init__(self, root, split, shuffle=False, transform=None):
        self.root = root
        self.split = split
        self.shuffle = shuffle
        self.transform = transform
        self.labels()
        self.filelist = self.get_filelist()

    def get_filelist(self):
        dir_path = join(self.root, "images", self.split)
        labels = self.label_to_class.keys()
        #sub_paths = [join(dir_path, sub) for sub in os.listdir(dir_path) if isdir(join(dir_path, sub))]
        sub_paths = [join(dir_path, sub) for sub in labels if isdir(join(dir_path, sub))]
        
        filelist = []
        for sub_path in sub_paths:
            files = [ join(sub_path, _file)  for _file in os.listdir(sub_path) if isfile(join(sub_path, _file))]
            filelist += files
        if self.shuffle == True:
            shuffle(filelist)
        return filelist

    def get_specific_labels(self, labels):
        dir_path = join(self.root, "images", self.split)
        sub_paths = [join(dir_path, sub) for sub in labels if isdir(join(dir_path, sub))]

        filelist = []
        for sub_path in sub_paths:
            files = [ join(sub_path, _file)  for _file in os.listdir(sub_path) if isfile(join(sub_path, _file))]
            filelist += files
        if self.shuffle == True:
            shuffle(filelist)
        return filelist        

    def labels(self, selected_species=None):
        # example.
        # label : "1355868" 
        # name : "Lactuca virosa L."
        # class : 0
        with open(join(self.root, "plantnet300K_species_id_2_name.json"), 'r') as file:
            label_to_name = json.load(file) #label => name
        if selected_species is not None:
            label_to_name = dict([(k,v) for k,v in label_to_name.items() if v in selected_species])
                        
        label_to_class = {} # label => class
        class_to_name = {} # class => name
        name_to_label = {} # name => label
        
        for label, name in zip(label_to_name.keys(), label_to_name.values()):
            class_to_name[len(label_to_class)] = name
            label_to_class[label] = len(label_to_class)
            name_to_label[name] = label
        
        self.class_to_name = class_to_name
        self.label_to_class = label_to_class
        self.name_to_label = name_to_label
        
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

class HierarchicalPlantNet(PlantNet):
    def __init__(self, root, split, fine_to_coarse, shuffle=False, transform=None):
        super().__init__(root, split, shuffle, transform)
        self.fine_to_coarse = fine_to_coarse

    def __getitem__(self, idx):
        img_path = self.filelist[idx]
        image = cv2.imread(img_path)
        if image is None:
            print(f"Fail to load image file : {img_path}")
        image = image[:,:,::-1]

        label = img_path.split('/')[-2]
        fine_class_id = self.label_to_class[label]
        coarse_class_id = self.fine_to_coarse[str(fine_class_id)]
        
        if self.transform: #transform must be albumentation's tranform.
            image = self.transform(image=image)["image"]
        return image, [coarse_class_id, fine_class_id]

class GeneraPlantNet(PlantNet):
    def __init__(self, root, split, shuffle=False, transform=None):
        super().__init__(root, split, shuffle, transform)

    def labels(self):
        with open(join(self.root, "plantnet300K_species_id_2_name.json"), 'r') as file:
            label_to_species = json.load(file) #label => name
        with open(join("data/plantnet", "species_to_fine.json"), 'r') as file:
            species_to_fine = json.load(file) #label => name
        with open(join("data/plantnet", "fine_to_coarse.json"), 'r') as file:
            fine_to_coarse = json.load(file) #label => name
        with open(join("data/plantnet", "genera_to_coarse.json"), 'r') as file:
            genera_to_coarse = json.load(file) #label => name
        coarse_to_genera = dict([(value, key) for key, value in genera_to_coarse.items()])
        
        label_to_class = {} # label => class
        class_to_name = {} # class => name
        name_to_label = {} # name => label        
                
        for label, species in zip(label_to_species.keys(), label_to_species.values()):
            coarse = fine_to_coarse[str(species_to_fine[species])]
            genera = coarse_to_genera[coarse]
            label_to_class[label] = coarse
            class_to_name[coarse] = genera
            name_to_label[genera] = label
        
        return label_to_class, class_to_name, name_to_label

class MiniPlantNet(PlantNet):
    def __init__(self, root, split, shuffle=False, transform=None, minimum_samples=32):
        super().__init__(root, split, shuffle=False, transform=None)
        self.minimum = minimum_samples

    def labels(self, selected_species=None): #change labels to new target classes
        dir_path = join(self.root, "images", "train")
        sub_paths = [join(dir_path, sub) for sub in os.listdir(dir_path) if isdir(join(dir_path, sub))]
        target_labels = [os.path.basename(sub_path) for sub_path in sub_paths if len(os.listdir(sub_path))>=self.minimum]

        with open(join(self.root, "plantnet300K_species_id_2_name.json"), 'r') as file:
            label_to_name = json.load(file) #label => name
        if selected_species is not None:
            label_to_name = dict([(k,v) for k,v in label_to_name.items() if v in selected_species])
            
        label_to_class = {} # label => class
        class_to_name = {} # class => name
        name_to_label = {} # name => label

        for label, name in zip(label_to_name.keys(), label_to_name.values()):
            if label not in target_labels: 
                continue
            class_to_name[len(label_to_class)] = name
            label_to_class[label] = len(label_to_class)
            name_to_label[name] = label
        return label_to_class, class_to_name, name_to_label     

    
class HierarchicalMiniPlantNet(MiniPlantNet):
    def __init__(self, root, split, shuffle=False, transform=None, minimum_samples=32):
        super().__init__(root, split, shuffle, transform, minimum_samples)
        with open(join("data/mini_plantnet", "fine_to_coarse.json"), 'r') as st_json:
                fine_to_coarse = json.load(st_json)
        self.fine_to_coarse = fine_to_coarse

    def __getitem__(self, idx):
        img_path = self.filelist[idx]
        image = cv2.imread(img_path)
        if image is None:
            print(f"Fail to load image file : {img_path}")
        image = image[:,:,::-1]

        label = img_path.split('/')[-2]
        fine_class_id = self.label_to_class[label]
        coarse_class_id = self.fine_to_coarse[str(fine_class_id)]
        
        if self.transform: #transform must be albumentation's tranform.
            image = self.transform(image=image)["image"]
        return image, [coarse_class_id, fine_class_id]
    

