from experiment.run_efficientB4 import *

exp_list = [exp_set1, exp_set2, exp_set3, exp_set4, exp_set5]

for exp_num, exp in enumerate(exp_list):
    exp_dir = f"/home/files/experiments/efficientB4/exp_set{exp_num+1}"
    weight_dir = f"{exp_dir}/checkpoints/checkpoint.pt"
    log_dir = f"{exp_dir}/logs/"
    exp(weight_dir, log_dir)