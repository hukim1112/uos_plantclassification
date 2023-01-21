import sys
import os
sys.path.append("/home/files/uos_plantclassification")
from flask import Flask, render_template
import albumentations as A
from albumentations.pytorch import ToTensorV2
from data import get_plantnet
from torch import nn
import cv2
from models import HierarchicalClassifier, EfficientB4


app = Flask(__name__)
host_addr = "0.0.0.0"
host_port = 8888

# #imagenet_class_index = json.load(open('imagenet_class_index.json'))
# data_loaders, class_to_name = get_plantnet(transforms=None) #get PlantNet-300K dataset by default options.
# model = EfficientB4(num_classes=len(class_to_name), loss_fn=nn.CrossEntropyLoss()) #get your model
# model.load("/home/files/experiments/mini_plantnet/efficientB4/exp2/checkpoints/checkpoint.pt") # load Its the best checkpoint.
# model.eval()

# def preprocess(image):
#     image_transform = A.Compose([
#                 A.LongestMaxSize(max_size=500),
#                 A.PadIfNeeded(min_height=int(380),
#                 min_width=int(380),
#                 position='top_left',
#                 border_mode=cv2.BORDER_CONSTANT),
#                 A.CenterCrop(380,380,p=1.0),
#                 A.Normalize(mean=0.0, std=1.0),
#                 ToTensorV2()])
#     return image_transform(image=image)["image"]


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    return render_template('index.html')

@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    return "hello"

if __name__ == '__main__':
    app.run(debug=True, host=host_addr, port=host_port)