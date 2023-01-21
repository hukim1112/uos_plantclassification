import json
from os.path import join
from os import listdir
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config.path import PATH
from data import get_mini_plantnet, MiniPlantNet
import pandas as pd

   
def make_df(df1_path, df2_path, name_clusters_path, species_to_num_samples_path):
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)
    with open(species_to_num_samples_path, 'r') as file:
        name_to_train_samples = json.load(file)

    with open(name_clusters_path, 'r') as file:
        name_clusters = json.load(file)
        
    size_of_cluster = {}
    proportion_in_cluster = {}
    cluster_id = {}
    
    _id = 0
    for cluster_member in list(name_clusters.values()):
        _id+=1
        total_samples_in_cluster = 0
        for member in cluster_member:
            size_of_cluster[member] = len(cluster_member)
            cluster_id[member] = _id
            total_samples_in_cluster+=name_to_train_samples[member]
        for member in cluster_member:
            proportion_in_cluster[member] = name_to_train_samples[member]/total_samples_in_cluster       

    train_samples = []  
    proportion = []
    family_size = []
    id_list = []
    for name in df1.name:
        train_samples.append(name_to_train_samples[name])
        proportion.append(proportion_in_cluster[name])
        family_size.append(size_of_cluster[name])
        id_list.append(cluster_id[name])

    df1['train_samples'] = train_samples
    df1['test_samples'] = df1["samples_per_class"]
    df1['proportion'] = proportion
    df1['family_size'] = family_size
    df1['cluster_id'] = id_list
    df3 = df1[['name', 'train_samples', 'test_samples', 'family_size', 'proportion', 'cluster_id', 'recall']]
    df3['recall_gap'] = df2['recall'] - df1['recall']
    return df3



'''
import json
from os.path import join
from os import listdir
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config.path import PATH
from data import get_mini_plantnet, MiniPlantNet
import pandas as pd


transforms = {
    'train': A.Compose([
        A.Resize(height=380, width=380),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()]),
    'val': A.Compose([
        A.Resize(height=380, width=380),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()]),
    'test': A.Compose([
        A.Resize(height=380, width=380),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()])
}
    
dataset = MiniPlantNet(root=PATH["PLANTNET-300K"], split="train", shuffle=False, transform=transforms["test"])

#mini plantnet의 학습데이터 수
label_to_train_samples = {}
name_to_train_samples = {}
for name, label in zip(dataset.name_to_label.keys(), dataset.name_to_label.values()):
    num_samples = len(listdir(join(PATH["PLANTNET-300K"], "images", "train", label)))
    label_to_train_samples[label] = num_samples
    name_to_train_samples[name] = num_samples
with open("/home/files/uos_plantclassification/data/mini_plantnet/name_to_train_samples.json", 'w', encoding='utf-8') as file:
    json.dump(name_to_train_samples, file, ensure_ascii=False, indent=2)

'''