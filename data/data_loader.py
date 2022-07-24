from config.path import PATH
from .PlantNet import PlantNet300K
from torchvision import transforms
from torch.utils.data import DataLoader

default_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()])
}

def get_plantnet(transforms=None):
    if transforms is None:
        transforms = default_transforms
        
    splits = ["train", "val", "test"]
    data_loaders = {}
    for split in splits:
        dataset = PlantNet300K(root=PATH["PLANTNET-300K"], split=split, shuffle=(split=="train"), transform=transforms[split])
        if split == "train":
            class_to_name = dataset.class_to_name
        data_loaders[split] = DataLoader(dataset, batch_size=64, shuffle=(split=="train"), num_workers=4)
    return data_loaders, class_to_name