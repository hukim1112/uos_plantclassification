from experiment.run import exp_set5

exp_dir = "/home/files/experiments/exp5"
weight_dir = f"{exp_dir}/checkpoints/checkpoint.pt"
log_dir = f"{exp_dir}/logs/"

exp_set5(weight_dir, log_dir)