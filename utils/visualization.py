import torch
import os
from matplotlib import pyplot as plt
import sys
import warnings
warnings.filterwarnings('ignore')
from torchvision import models
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM,LayerCAM,GradCAMElementWise,XGradCAM,AblationCAM,ScoreCAM, \
    EigenCAM,EigenGradCAM,FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image
from PIL import Image
import random
import torchvision.transforms.functional as TF
from torchvision import transforms


def bring_imgs(path,img_num): #이미지가 들어있는 경로와 가져오고 싶은 이미지의 갯수를 인풋으로 받음
    
    img_list=os.listdir(path) #이미지가 들어있는 경로 내의 모든 사진을 받아옴
    if img_num > len(img_list):
        print('이미지의 갯수가 입력값보다 적어서 폴더 내 모든 사진을 받아옵니다. 사진 갯수: {0}'.format(len(img_list)))
        img_num=len(img_list)
    image_path = np.array(random.sample(img_list,img_num)) # 해당 카테고리 사진들중 랜덤하게 img_num개 뽑아옴
    images=[0]*img_num

    for i in range(img_num):
        images[i]=os.path.join(path,image_path[i])
        img = np.array(Image.open(images[i]))
        img = cv2.resize(img, (256, 256))
        img = np.float32(img) / 255
        images[i]=img
        
    return np.array(images) #(256,256,3) 인 이미지들들을 리스트로 묶어서 반환시킴

def path2imgs(paths): #주어진 이미지 경로들을 이미지로 읽어옴.
    images = []
    for path in paths:
        img = np.array(Image.open(path))
        img = cv2.resize(img, (256, 256))
        img = np.float32(img) / 255
        images.append(img)
    return np.array(images) #(256,256,3) 인 이미지들들을 리스트로 묶어서 반환시킴



class ShapeError(Exception):
    def __str__(self):
        return "테이블 크기와 입력 이미지 갯수가 맞지 않습니다"


# 여러개의 사진을 테이블 형식으로 묶어서 하나의 이미지로 만들어주는 함수
def make_table(images,shape): #images 는 이미지들 담겨있는 리스트임
    """1. 행개수씩 이미지 리스트를 뽑아서 image들 hstack 해줌 (가로사이즈가 5면 인덱스를 0~4, 5~9 ... 가져옴)
       2. hstack 한 이미지를 row_list에 담아줌  
       3. 담은 row이미지들을 튜플로 묶어서 vstack 해줌"""
    #image_table=images[0]
    if (shape[0])*(shape[1])!=len(images):
        raise(ShapeError)

    try:    
        row_list=[]

        for row_idx in range(shape[0]): #row_idx 는 0,1,2
            image_table=images[row_idx*shape[1]] #이러면 images[0] [5] [10] 가져옴 
            #맨처음 받는 image_table 이 행의 앞단이니까 여기에 원본 사진을 받아주면된다.
            #
            for col_idx in range((row_idx*shape[1])+1,(row_idx+1)*(shape[1])):
                image_table=np.hstack((image_table, images[col_idx]))
            row_list.append(image_table)
        image_table=np.vstack(tuple(row_list))
        #plt.imshow(image_table)
        #plt.show()
        return image_table
    
    except ShapeError as e:
        print(e)       
        
#종류별 CAM사진을 리스트로 묶어주는 함수 
def printCAM(label, images, model, target_layers): #images: (256,256,3) 사이즈의 샘플이미지들 묶음(np형)
#입력 레이어가 모델마다 다른 점을 감안하여 입력에 추가하여 사용자가 직접 입력하도록 변경해주었습니다
    
    targets = [ClassifierOutputTarget(label)] 
    #target_layers = [model.features]
    cam_list=[]
    for img in images:
        cam_list.append(img)
        input_tensor = preprocess_image(img,
                                      mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_tensor=input_tensor.to(device)
        
        method=(GradCAM(model=model, target_layers=target_layers),LayerCAM(model=model, target_layers=target_layers),
                EigenCAM(model=model, target_layers=target_layers),EigenGradCAM(model=model, target_layers=target_layers))
        
        for i in range(len(method)):
            with method[i] as cam:
                grayscale_cams = cam(input_tensor=input_tensor, targets=targets) 
                cam_out = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
                cam_out = cam_out.astype(float) / 255
            cam_list.append(cam_out)
    
    return np.array(cam_list) #이미지들 담긴 리스트 리턴


#원본, 흑백캠,칼라캠, 원본+캠, 바운딩 박스 사진들을 리스트로 묶어주는 함수
def diverse_CAM(label,images,model,CAMname,target_layers): #images: (256,256,3) 사이즈의 샘플이미지들 묶음(np형)
#입력 레이어가 모델마다 다른 점을 감안하여 입력에 추가하여 사용자가 직접 입력하도록 변경해주었습니다
    targets = [ClassifierOutputTarget(label)] 
    #target_layers = [model.features]
    img_list=[]
    for img in images:
        img_list.append(img)
        img2=img.copy() #나중에 바운딩 박스 표시할 이미지

        input_tensor = preprocess_image(img,
                                      mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input_tensor=input_tensor.to(device)
        
        method={'gradCAM':GradCAM(model=model, target_layers=target_layers),
                'layerCAM':LayerCAM(model=model, target_layers=target_layers),
                'eigenCAM':EigenCAM(model=model, target_layers=target_layers),
                'eigengradCAM':EigenGradCAM(model=model, target_layers=target_layers)}
        
        with method[CAMname] as cam:
            grayscale_cams = cam(input_tensor=input_tensor, targets=targets) 
            cam_out = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
        black_cam=cv2.merge([grayscale_cams[0,:],grayscale_cams[0,:],grayscale_cams[0,:]])
        cam = np.uint8(255*grayscale_cams[0,:]) #흑백 캠을 표시해줌 (256,256)
        colorcam=cv2.applyColorMap(np.uint8(255*black_cam), cv2.COLORMAP_JET)

        colorcam=cv2.cvtColor(colorcam,cv2.COLOR_BGR2RGB)

        thresh = cv2.threshold(cam, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] #OTSU알고리즘으로 임계값 찾음
        
         #Find contours
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #외곽선을 검출해줌
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(cam_out, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.rectangle(img2, (x, y), (x + w, y + h), (1,0.9,0), 2)
            
        img_list.append(black_cam)
        img_list.append(colorcam.astype(float) / 255)
        img_list.append(cam_out.astype(float) / 255)
        img_list.append(img2.astype(float))
        
        #print(img2)
    return np.array(img_list) #이미지들 모은 리스트 리턴
