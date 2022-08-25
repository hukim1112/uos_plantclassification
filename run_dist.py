import numpy as np
import torch
import pandas as pd
from torch import nn
from os.path import join
from config.path import PATH
from embeddings.extract_embeddings import Embedder
from embeddings.set_distance import SetDist
from utils.visualization import path2imgs, make_table
from matplotlib import pyplot as plt

def filter_category(df):
    df = df.dropna()
    df = df[df['samples_per_class']>=32]
    df['f1'] = df.apply(lambda x: 2 * (x['precision'] * x['recall']) / (x['precision'] + x['recall']+0.00001), axis=1)
    df = df[df['f1']>0.5]
    df = df.sort_values('f1', ascending=False)
    return df


root = PATH["PLANTNET-300K"]
split = 'train'
emb = Embedder(root, split)
labels = list(emb.label_to_class.keys())
categorical_metrics = pd.read_csv("/home/files/experiments/efficientB4/exp_set3/logs/categorical_metrics_test.csv")
filtered_cat = filter_category(categorical_metrics)

name_to_label = {}

for l,n in zip(emb.label_to_name.keys(), emb.label_to_name.values()):
    name_to_label[n] = l
labels = [name_to_label[n] for n in filtered_cat.name]

num_label = len(labels)
dist_matrix = np.zeros(shape=(num_label, num_label))-1

dist = SetDist()
stats = {}
for label in labels:
    embeddings, top_1_prob, correctness, file_paths = emb.load_embeddings(join("/home/files/experiments/plantnet_embeddings_v3", split), label)
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
    np.save("label148_dist_matrix.npy", dist_matrix)

import pickle
with open("label148_list.txt", "wb") as fp:
    pickle.dump(labels, fp)