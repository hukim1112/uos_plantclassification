from models import EfficientB4, VGG19, ResNet101, WideResNet101_2
from experiment.mini_plantnet import baseline1, baseline2, deep_hierarchical_classifier, genera_species_hierarchical_classifier

def train_EfficientB4(device):
    exp_list = [baseline1, baseline2]
    exp_name = ["exp_set1", "exp_set2"]
    for exp, name in zip(exp_list, exp_name):
        exp_dir = f"/home/files/experiments/mini_plantnet/baseline/EfficientB4/{name}"
        exp(exp_dir, EfficientB4, device)

def train_hierarchical_EfficientB4(device):
    exp_dir = f"/home/files/experiments/mini_plantnet/baseline/EfficientB4/exp_set2"
    for cluster_radius in [25, 30, 35]:
        deep_hierarchical_classifier(exp_dir, EfficientB4, cluster_radius, device, finetune=False)

def train_genera_species_hierarchical_classifier(device, method):
    exp_dir = f"/home/files/experiments/mini_plantnet/genera_species_hierarchical_classifier"
    baseline_weight_path = "/home/files/experiments/mini_plantnet/baseline/EfficientB4/exp_set2/checkpoints/checkpoint.pt"
    genera_species_hierarchical_classifier(exp_dir, EfficientB4, device, baseline_weight_path, method)

def train_VGG19(device):
    exp_list = [baseline1, baseline2]
    exp_name = ["exp_set3", "exp_set4"]
    for exp, name in zip(exp_list, exp_name):
        exp_dir = f"/home/files/experiments/mini_plantnet/baseline/VGG19/{name}"
        exp(exp_dir, VGG19, device)
        
def train_ResNet101(device):
    exp_list = [baseline1, baseline2]
    exp_name = ["exp_set1", "exp_set2"]
    for exp, name in zip(exp_list, exp_name):
        exp_dir = f"/home/files/experiments/mini_plantnet/baseline/VGG19/{name}"
        exp(exp_dir, ResNet101, device)

def train_WideResNet101_2(device):
    exp_list = [baseline1, baseline2]
    exp_name = ["exp_set1", "exp_set2"]
    for exp, name in zip(exp_list, exp_name):
        exp_dir = f"/home/files/experiments/mini_plantnet/baseline/VGG19/{name}"
        exp(exp_dir, WideResNet101_2, device)