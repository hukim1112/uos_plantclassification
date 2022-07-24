from experiment.run import exp_set1

exp_dir = "/home/files/experiments/exp1"
weight_dir = f"{exp_dir}/checkpoints/checkpoint.pt"
log_dir = f"{exp_dir}/logs/"

exp_set1(weight_dir, log_dir)