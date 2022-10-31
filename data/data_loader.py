from config.path import PATH
from .PlantNet import PlantNet, GeneraPlantNet, HierarchicalPlantNet, MiniPlantNet, HierarchicalMiniPlantNet
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

default_transforms = {
    'train': A.Compose([
        A.Resize(height=256, width=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()]),
    'val': A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()]),
    'test': A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()])
}

def get_plantnet(transforms=None, batch_size=32):
    if transforms is None:
        transforms = default_transforms
        
    splits = ["train", "val", "test"]
    data_loaders = {}
    for split in splits:
        dataset = PlantNet(root=PATH["PLANTNET-300K"], split=split, shuffle=(split=="train"), transform=transforms[split])
        if split == "train":
            class_to_name = dataset.class_to_name
        data_loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"), num_workers=4)
    return data_loaders, class_to_name

def get_genera_plantnet(transforms=None, batch_size=32):
    if transforms is None:
        transforms = default_transforms
        
    splits = ["train", "val", "test"]
    data_loaders = {}
    for split in splits:
        dataset = GeneraPlantNet(root=PATH["PLANTNET-300K"], split=split, shuffle=(split=="train"), transform=transforms[split])
        if split == "train":
            class_to_name = dataset.class_to_name
        data_loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"), num_workers=4)
    return data_loaders, class_to_name

def get_hierarchical_plantnet(transforms=None, batch_size=32):
    if transforms is None:
        transforms = default_transforms
        
    splits = ["train", "val", "test"]
    data_loaders = {}
    for split in splits:
        dataset = HierarchicalPlantNet(root=PATH["PLANTNET-300K"], split=split, shuffle=(split=="train"), transform=transforms[split])
        if split == "train":
            class_to_name = dataset.class_to_name
        data_loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"), num_workers=4)
    return data_loaders, class_to_name

def get_mini_plantnet(transforms=None, batch_size=32):
    if transforms is None:
        transforms = default_transforms
        
    splits = ["train", "val", "test"]
    data_loaders = {}
    for split in splits:
        dataset = MiniPlantNet(root=PATH["PLANTNET-300K"], split=split, shuffle=(split=="train"), transform=transforms[split])
        if split == "train":
            class_to_name = dataset.class_to_name
        data_loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"), num_workers=4)
    return data_loaders, class_to_name

def get_hierarchical_mini_plantnet(fine_to_coarse, transforms=None, batch_size=32):
    if transforms is None:
        transforms = default_transforms
        
    splits = ["train", "val", "test"]
    data_loaders = {}
    for split in splits:
        dataset = HierarchicalMiniPlantNet(root=PATH["PLANTNET-300K"], split=split, shuffle=(split=="train"), transform=transforms[split])
        if split == "train":
            class_to_name = dataset.class_to_name
        data_loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split=="train"), num_workers=4)
    return data_loaders, class_to_name