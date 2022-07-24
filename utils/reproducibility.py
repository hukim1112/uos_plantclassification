import random
import numpy as np
import torch

def set_seed(random_seed, use_gpu, dev = True, print_out=True):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if use_gpu:
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    if dev == False:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False