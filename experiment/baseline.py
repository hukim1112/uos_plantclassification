from models import EfficientB4, VGG19, ResNet101, WideResNet101_2
from experiment.train_settings import exp_set1, exp_set2

def train_EfficientB4(device):
    exp_list = [exp_set1, exp_set2]
    exp_name = ["exp_set1", "exp_set2"]
    for exp, name in zip(exp_list, exp_name):
        exp_dir = f"/home/files/experiments/mini_plantnet/baseline/EfficientB4/{name}"
        weight_dir = f"{exp_dir}/checkpoints/checkpoint.pt"
        log_dir = f"{exp_dir}/logs/"
        exp(weight_dir, log_dir, EfficientB4, device)

def train_VGG19(device):
    exp_list = [exp_set1, exp_set2]
    exp_name = ["exp_set3", "exp_set4"]
    for exp, name in zip(exp_list, exp_name):
        exp_dir = f"/home/files/experiments/mini_plantnet/baseline/VGG19/{name}"
        weight_dir = f"{exp_dir}/checkpoints/checkpoint.pt"
        log_dir = f"{exp_dir}/logs/"
        exp(weight_dir, log_dir, VGG19, device)
        
def train_ResNet101(device):
    exp_list = [exp_set1, exp_set2]
    exp_name = ["exp_set1", "exp_set2"]
    for exp, name in zip(exp_list, exp_name):
        exp_dir = f"/home/files/experiments/mini_plantnet/baseline/ResNet101/{name}"
        weight_dir = f"{exp_dir}/checkpoints/checkpoint.pt"
        log_dir = f"{exp_dir}/logs/"
        exp(weight_dir, log_dir, ResNet101, device)

def train_WideResNet101_2(device):
    exp_list = [exp_set1, exp_set2]
    exp_name = ["exp_set1", "exp_set2"]
    for exp, name in zip(exp_list, exp_name):
        exp_dir = f"/home/files/experiments/mini_plantnet/baseline/WideResNet101_2/{name}"
        weight_dir = f"{exp_dir}/checkpoints/checkpoint.pt"
        log_dir = f"{exp_dir}/logs/"
        exp(weight_dir, log_dir, WideResNet101_2, device)

'''
def multi_run():
    exp_list = [exp_set3, exp_set5]
    exp_name = ["exp_set3", "exp_set5"]

    for exp, name in zip(exp_list, exp_name):
        exp_dir = f"/home/files/experiments/efficientB4_augment/{name}"
        weight_dir = f"{exp_dir}/checkpoints/checkpoint.pt"
        log_dir = f"{exp_dir}/logs/"
        exp(weight_dir, log_dir)
'''