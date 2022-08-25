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
device = "cuda:2"
weight_dir = "/home/files/experiments/efficientB4/exp_set3/checkpoints/checkpoint.pt"


from os.path import join
for split in ["train", "val", "test"]:
    emb = Embedder(root, split)
    labels = list(emb.label_to_class.keys())
    model = EfficientB4(num_classes=emb.num_classes, loss_fn=nn.CrossEntropyLoss()) #get your model
    model.load(weight_dir) # load Its the best checkpoint.
    emb.get_model(model, transform, device)
    for label in tqdm(labels):
        emb.save_embeddings(join("/home/files/experiments/plantnet_embeddings_v3", split), label)