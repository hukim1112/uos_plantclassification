import json
import os

#식물 이름 던지면-> path 가져오고, 변환 이름 가져오고, 라벨 가져오고 
#식물 이름 던졌을 때 {라벨:이름임} -> 키벨류 바꾸면 해결가능
class PlantPath:
    def __init__(self):
        with open("/home/files/datasets/plantnet_300K/plantnet300K_species_id_2_name.json", 'r') as f:
            self.json_data = json.load(f)
        self.root_dir="/home/files/datasets/plantnet_300K/images/"

    def bring_path(self, plantname=None, kind='test'): #bring the path of img folder's path 
        
        root_dir=os.path.join(self.root_dir, kind)
        data_reversed = {v:k for k,v in self.json_data.items()}
        if plantname in data_reversed.keys():
            label=data_reversed.get(plantname)
            path=os.path.join(root_dir,label)
            print("bring path in {} data of {}".format(kind,plantname))
            return path
        else:
            print("there's no plantname on. select the plant\n")
            for i,(k,v) in enumerate(self.json_data.items(),0):
                path=os.path.join(self.root_dir,'train',k)
                print(i, "-", k,":",v,"(num of data(train): {})".format(len(os.listdir(path))))
            return
    
    def bring_categoriesWithlabels(self): #bring total categories' name of data by sorted list(it's index is same as trained model's labeling method)
        
        root_dir=os.path.join(self.root_dir, 'test')
        labels_list=[sub for sub in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, sub))]
        labels_list=sorted(labels_list)
        categories_list=[0]*len(labels_list)
        for i in range(len(labels_list)):
            if labels_list[i] in list(self.json_data.keys()):
                categories_list[i]=self.json_data[labels_list[i]]
        
        return categories_list,labels_list
    
    def find_label(self,plantname=None): #plantname-> plant label of PlantNet300k
        plant_label=None
        if plantname is not None:
            for item in self.json_data.items():
                if item[1]==plantname:
                    plant_label=item[0]
                    return plant_label
        else:
            print("there's no plantname on. select the plant\n")
            for i,(k,v) in enumerate(self.json_data.items(),0):
                print(i, ": ", k,"-",v)
            return
    
    def __call__(self, plantname=None, kind='test'):
        return self.bring_path(plantname,kind) 