from .ImageClassifier import Baseline
import os
import torch
from torch import nn
import timm

class VGG19(Baseline):
    def __init__(self, num_classes, loss_fn, fc_type="shallow"):
        super(VGG19, self).__init__(loss_fn)
        self.num_classes = num_classes
        self.fc_type = fc_type
        self.patch_embedding = self.get_patch_embedding()
        self.embedding = self.get_embedding()
        self.fc = self.get_fc()     

    def get_patch_embedding(self):
        cnn = timm.create_model('vgg19', pretrained=False)
        return nn.Sequential( *list(cnn.children())[:-2])
    
    def get_embedding(self):
        return nn.Sequential(nn.Flatten())

    def get_fc(self):
        if self.fc_type == 'deep':
            fc = nn.Sequential(nn.Linear(61952, 1024),
                                        nn.BatchNorm1d(1024,  momentum=0.1),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.BatchNorm1d(1024,  momentum=0.1),
                                        nn.ReLU(),
                                        nn.Linear(1024, self.num_classes))

        elif self.fc_type == 'shallow':
            fc = nn.Linear(61952, self.num_classes)
        else:
            raise ValueError(f"Wrong fc-type input {self.fc_type}")
        return fc

