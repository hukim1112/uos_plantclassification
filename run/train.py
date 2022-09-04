from experiment.run_efficientB4_mini_plantnet import *
def run1():
    name="exp1"
    exp_dir = f"/home/files/experiments/mini_plantnet/efficientB4/{name}"
    weight_dir = f"{exp_dir}/checkpoints/checkpoint.pt"
    log_dir = f"{exp_dir}/logs/"
    exp_set1(weight_dir, log_dir)
def run2():
    name="exp2"
    exp_dir = f"/home/files/experiments/mini_plantnet/efficientB4/{name}"
    weight_dir = f"{exp_dir}/checkpoints/checkpoint.pt"
    log_dir = f"{exp_dir}/logs/"
    exp_set2(weight_dir, log_dir)
def run3():
    name="exp3"
    exp_dir = f"/home/files/experiments/mini_plantnet/efficientB4/{name}"
    weight_dir = f"{exp_dir}/checkpoints/checkpoint.pt"
    log_dir = f"{exp_dir}/logs/"
    exp_set3(weight_dir, log_dir)


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