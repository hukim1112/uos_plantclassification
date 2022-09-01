import torch 
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import json
from matplotlib import pyplot as plt
from utils import visualization as vi
from torch import nn
import os


class Inferencer(): #사진에 대한 모델의 예측 top5를 바로 찍어주는 클래스를 만들자
    
    def __init__(self, model, categories,path,device="cuda:0", dataset='plantnet'): #categories is total sorted categories' list of data
        self.model=model
        self.categories=categories
        self.device=device
        self.dataset=dataset
        self.path=path 
        if dataset=='plantnet': #플랜트넷의 경우 카테고리 리스트 가져올 때 정렬하고 가져와야함
            label=path.split('/')[-2] # 플랜트넷의 경우 숫자로 카테고리 되어있어서 변환해야함
        
        else:
            label=path.split('/')[-3] # 리빙랩일 경우 라벨이 한글
            
        self.idx=self.categories.index(label) # 학습시의 데이터 인덱스 넘버
    
        with open("/home/files/datasets/plantnet_300K/plantnet300K_species_id_2_name.json", 'r') as f:
            self.json_data = json.load(f)
            
    def transform(self):
        data_transforms = A.Compose([
        A.LongestMaxSize(max_size=500),
        A.PadIfNeeded(min_height=int(380),
        min_width=int(380),
        position='top_left',
        border_mode=cv2.BORDER_CONSTANT),
        A.CenterCrop(height=380, width=380, p=1.0),
        A.Normalize(mean=0.0, std=1.0),
        ToTensorV2()])
        return data_transforms
    
    def file(self, path=None): #이미지 path를 넣어 이미지 가져옴('/home/files/datasets/plantnet_300K/images/test/[label]/[img name]') 
        
        if path==None: # path가 None 이면 input img
            image = np.array(Image.open(self.path))
        else:
            image = np.array(Image.open(path))
        if image.shape[2]>3:
            image=image[...:3]
        
        transform=self.transform()
        image = transform(image=image)['image']
        
        if path==None:
            plt.imshow(image.permute(1,2,0))
            plt.show()
        
        image = image.unsqueeze(0)
        
        return image #(1,3,256,256) size

    def __call__(self,X,with_photo=True): #객체 호출하면 바로 예측 top 5 확률값과  값이 나오도록
        
        if type(X)!=torch.Tensor: #넘파이로 이미지 넘기면 Albumentation 변환 따로 적용 
            transform=self.transform()
            X=transform(image=X)['image']
            X=X.unsqueeze(0)
            
        self.model.eval()    
        with torch.no_grad():
            pred=self.model(X.to(self.device))
            print(pred.shape)
            prob=nn.Softmax(dim=1)
            pred=prob(pred)
            predicted_top5=torch.topk(pred,5)    #한 배치의 top5정보 지님, 형태는 [value= batch size*5 형태, indices=b size*5 형태]
            batch_scores=predicted_top5[0]
            batch_indices=predicted_top5[1]
        top5_dict={}
        
        if self.dataset=='plantnet':
            name_list=self.label2name()
            top5_dict[name_list[self.idx]]=[(batch_scores[0][0].item(),name_list[batch_indices[0][0].item()]),
                                        (batch_scores[0][1].item(),name_list[batch_indices[0][1].item()]),
                                        (batch_scores[0][2].item(),name_list[batch_indices[0][2].item()]),
                                        (batch_scores[0][3].item(),name_list[batch_indices[0][3].item()]),
                                        (batch_scores[0][4].item(),name_list[batch_indices[0][4].item()])]
        else:
            top5_dict[self.categories[self.idx]]=[(batch_scores[0][0].item(),self.categories[batch_indices[0][0].item()]),
                                        (batch_scores[0][1].item(),self.categories[batch_indices[0][1].item()]),
                                        (batch_scores[0][2].item(),self.categories[batch_indices[0][2].item()]),
                                        (batch_scores[0][3].item(),self.categories[batch_indices[0][3].item()]),
                                        (batch_scores[0][4].item(),self.categories[batch_indices[0][4].item()])]
        if with_photo==True:
            CAMlabel_list=[]
            out_imgs=[]
            for k in range(5):
                CAMlabel_list.append(self.categories[batch_indices[0][k].item()])

            for label in CAMlabel_list:
                #img_list=os.listdir(os.path.join('/home/files/datasets/plantnet_300K/images/test/',label))
                #sample=random.sample(img_list,1)[0]
                #sample_path=os.path.join('/home/files/datasets/plantnet_300K/images/test/',label,sample)
                #sample_img=self.file(sample_path)[0]
                #out_imgs.append(sample_img)
                sample_img=vi.bring_imgs(os.path.join('/home/files/datasets/plantnet_300K/images/test/',label),1)[0]
                out_imgs.append(sample_img)
                self.visualize_pred(sample_img)
                
        return top5_dict # top5 acc 딕셔너리를 출력해주면된다. 
    
    def label2name(self):
        #카테고리명이 라벨로 되어있는 경우 식물이름으로 변환시켜주기
        name_list=[]
        for label in self.categories:
            plantname=self.json_data[label]
            name_list.append(plantname)

        return name_list

    def visualize_pred(self,imgs): #top5 에 대한 visualize 하는거임
        CAMimgs=vi.diverse_CAM(self.idx,imgs,self.model,CAMname='eigenCAM',target_layers=[self.model.patch_embedding])
        shape=(len(imgs),5)
        vstable=vi.make_table(CAMimgs,shape=shape)
        plt.imshow(vstable)

        word=['original','CAM(gray)','CAM(color)','eigenCAM','detection box']
        for i in range(5):
            plt.text(256*i,-10,word[i])
        plt.show()
