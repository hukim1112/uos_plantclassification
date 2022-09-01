from experiment.run_efficientB4_mini_plantnet import *

name="exp1"
exp_dir = f"/home/files/experiments/mini_plantnet/efficientB4/{name}"
weight_dir = f"{exp_dir}/checkpoints/checkpoint.pt"
log_dir = f"{exp_dir}/logs/"
exp_set1(weight_dir, log_dir)