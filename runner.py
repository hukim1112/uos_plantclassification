from run.train import run1, run2, run3
from run.embeddings import extract_embeddings, calculate_dist_matrix

#extract_embeddings
import albumentations as A
from albumentations.pytorch import ToTensorV2
exp_path = "/home/files/experiments/mini_plantnet/efficientB4/exp1"
transforms = A.Compose([
        A.Resize(height=380, width=380),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()])
device='cuda:0'
extract_embeddings(exp_path, transforms, device)
calculate_dist_matrix(exp_path)