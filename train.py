#from experiment.training import train_EfficientB4, train_ResNet101, train_WideResNet101_2, train_VGG19, train_genera_species_hierarchical_classifier
from models import ResNet101, WideResNet101_2
from experiment.plantnet import genera

exp_dir = f"/home/files/experiments/plantnet/baseline/ResNet101/genera"
device="cuda:2"
genera(exp_dir, ResNet101, device)

exp_dir = f"/home/files/experiments/plantnet/baseline/WideResNet101_2/genera"
device="cuda:2"
genera(exp_dir, WideResNet101_2, device)