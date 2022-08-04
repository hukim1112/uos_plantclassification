from experiment.run import exp_set3, exp_set4

exp_dir = "/home/files/experiments/exp3"
weight_dir = f"{exp_dir}/checkpoints/checkpoint.pt"
log_dir = f"{exp_dir}/logs/"

exp_set3(weight_dir, log_dir)

exp_dir = "/home/files/experiments/exp4"
weight_dir = f"{exp_dir}/checkpoints/checkpoint.pt"
log_dir = f"{exp_dir}/logs/"

exp_set4(weight_dir, log_dir)