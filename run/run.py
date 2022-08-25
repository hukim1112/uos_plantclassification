from experiment.run_efficientB4 import *

exp_list = [exp_set3, exp_set5]
exp_name = ["exp_set3", "exp_set5"]

for exp, name in zip(exp_list, exp_name):
    exp_dir = f"/home/files/experiments/efficientB4_augment/{name}"
    weight_dir = f"{exp_dir}/checkpoints/checkpoint.pt"
    log_dir = f"{exp_dir}/logs/"
    exp(weight_dir, log_dir)