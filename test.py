from experiment.mini_plantnet import test
from models import EfficientB4, VGG19, ResNet101, WideResNet101_2

device='cuda:1'
exp_dir="/home/files/experiments/mini_plantnet/baseline/EfficientB4/exp_set2"
MODEL=EfficientB4
test(exp_dir, MODEL, device)