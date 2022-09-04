import pickle, json
import numpy as np
import torch
import pandas as pd
from torch import nn
from os.path import join
from os import listdir
from config.path import PATH
from embeddings.extract_embeddings import Embedder
from embeddings.set_distance import SetDist
from utils.visualization import path2imgs, make_table
from matplotlib import pyplot as plt
import sys
sys.path.append("/home/files/uos_plantclassification")
split = 'train'
root = PATH["PLANTNET-300K"]
emb_path = "/home/files/experiments/plantnet_embeddings_v3"
metric_path = "/home/files/experiments/efficientB4/exp_set3/logs/categorical_metrics_test.csv"

'''
def filter_category(df):
    df = df.dropna()
    df = df[df['samples_per_class']>=32]
    df['f1'] = df.apply(lambda x: 2 * (x['precision'] * x['recall']) / (x['precision'] + x['recall']+0.00001), axis=1)
    df = df[df['f1']>0.5]
    df = df.sort_values('f1', ascending=False)
    return df
'''

def filter_category(minimum=32):
    dir_path = "/home/files/datasets/plantnet_300K/images/train/"
    _list = []
    for sub in listdir(dir_path):
        label = sub
        sub_path= join(dir_path, sub)
        sub_num = len(listdir(sub_path))
        if sub_num>=minimum:
            _list.append(label)
    return _list   


def calculate_dist_matrix():   
    emb = Embedder(root, split)
    labels = list(emb.label_to_class.keys())
    categorical_metrics = pd.read_csv(metric_path)
    labels = filter_category(minimum=100)

    # name_to_label = {}
    # for l,n in zip(emb.label_to_name.keys(), emb.label_to_name.values()):
    #     name_to_label[n] = l

    num_label = len(labels)
    dist_matrix = np.zeros(shape=(num_label, num_label))-1

    dist = SetDist()
    stats = {}
    for label in labels:
        print(f"calcuate stats : {label}")
        embeddings, top_1_prob, correctness, file_paths = emb.load_embeddings(join(emb_path, split), label)
        stats[label] = dist.multi_variate_gaussian(torch.tensor(embeddings))

    for i in range(dist_matrix.shape[0]):
        print(i)
        for j in range(dist_matrix.shape[1]):
            tmp = None
            label1 = labels[i]
            label2 = labels[j]
            if dist_matrix[i,j] != -1:
                tmp = dist_matrix[i,j]
            if dist_matrix[j,i] != -1:
                tmp = dist_matrix[j,i]

            if tmp is not None:
                dist_matrix[i,j] = tmp
            else:
                dist_matrix[i,j] = dist.bhattacharyya_from_stats(stats[label1], stats[label2])
        np.save(join(emb_path, f"label{num_label}_dist_matrix.npy"), dist_matrix)

    with open(join(emb_path, f"label{num_label}_list.txt"), "wb") as fp:
        pickle.dump(labels, fp)

def calculate_dists_each_label_from_matrix():
    emb = Embedder(root, split)
    labels = list(emb.label_to_class.keys())

    with open(join(emb_path, "label467_list.txt"), "rb") as fp:
        labels = pickle.load(fp)
    dist_matrix = np.load(join(emb_path, "label467_dist_matrix.npy"))

    label_to_name = emb.label_to_name
    name_to_label = name_to_label=dict([(value, key) for key, value in label_to_name.items()])

    dists_from_label = {}
    for i in range(dist_matrix.shape[0]):
        target_label = labels[i]
        target_dists = dist_matrix[i]
        sorted_idx = np.argsort(target_dists)
        sorted_dists = target_dists[sorted_idx]
        sorted_labels = np.array(labels)[sorted_idx]
        dists_from_label[target_label] = {"target_label" : target_label, "dists" : list(sorted_dists), "labels" : list(sorted_labels)}

    with open(join(emb_path, "dists_from_label467.json"), 'w', encoding='utf-8') as f: 
        json.dump(dists_from_label, f, ensure_ascii=False, indent=2)

def main():
    #calculate_dist_matrix()
    calculate_dists_each_label_from_matrix()
    pass

main()