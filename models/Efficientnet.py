from .ImageClassifier import Baseline
import os
import torch
from torch import nn
import timm

class EfficientB0(Baseline):
    def __init__(self, num_classes, loss_fn, fc_type="deep"):
        super(EfficientB0, self).__init__(loss_fn)
        self.num_classes = num_classes
        self.fc_type = fc_type
        self.patch_embedding = self.get_patch_embedding()
        self.embedding = self.get_embedding()
        self.fc = self.get_fc()     

    def get_patch_embedding(self):
        cnn = timm.create_model('efficientnet_b0', pretrained=False)
        return nn.Sequential( *list(cnn.children())[:-2])
    
    def get_embedding(self):
        return nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())

    def get_fc(self):
        if self.fc_type == 'deep':
            fc = nn.Sequential(nn.Linear(1280, 1024),
                                        nn.BatchNorm1d(1024,  momentum=0.1),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.BatchNorm1d(1024,  momentum=0.1),
                                        nn.ReLU(),
                                        nn.Linear(1024, self.num_classes))

        elif self.fc_type == 'shallow':
            fc = nn.Linear(1280, self.num_classes)
        else:
            raise ValueError(f"Wrong fc-type input {self.fc_type}")
        return fc

class EfficientB3(Baseline):
    def __init__(self, num_classes, loss_fn, fc_type="deep"):
        super(EfficientB3, self).__init__(loss_fn)
        self.num_classes = num_classes
        self.fc_type = fc_type
        self.patch_embedding = self.get_patch_embedding()
        self.embedding = self.get_embedding()
        self.fc = self.get_fc()     

    def get_patch_embedding(self):
        cnn = timm.create_model('efficientnet_b3', pretrained=False)
        return nn.Sequential( *list(cnn.children())[:-2])
    
    def get_embedding(self):
        return nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())

    def get_fc(self):
        if self.fc_type == 'deep':
            fc = nn.Sequential(nn.Linear(1536, 1024),
                                        nn.BatchNorm1d(1024,  momentum=0.1),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.BatchNorm1d(1024,  momentum=0.1),
                                        nn.ReLU(),
                                        nn.Linear(1024, self.num_classes))

        elif self.fc_type == 'shallow':
            fc = nn.Linear(1536, self.num_classes)
        else:
            raise ValueError(f"Wrong fc-type input {self.fc_type}")
        return fc

class EfficientB4(Baseline):
    def __init__(self, num_classes, loss_fn, fc_type="shallow"):
        super(EfficientB4, self).__init__(loss_fn)
        self.num_classes = num_classes
        self.fc_type = fc_type
        self.patch_embedding = self.get_patch_embedding()
        self.embedding = self.get_embedding()
        self.fc = self.get_fc()     

    def get_patch_embedding(self):
        cnn = timm.create_model('efficientnet_b4', pretrained=False)
        return nn.Sequential( *list(cnn.children())[:-2])
    
    def get_embedding(self):
        return nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())

    def get_fc(self):
        if self.fc_type == 'deep':
            fc = nn.Sequential(nn.Linear(1792, 1024),
                                        nn.BatchNorm1d(1024,  momentum=0.1),
                                        nn.ReLU(),
                                        nn.Linear(1024, 1024),
                                        nn.BatchNorm1d(1024,  momentum=0.1),
                                        nn.ReLU(),
                                        nn.Linear(1024, self.num_classes))

        elif self.fc_type == 'shallow':
            fc = nn.Linear(1792, self.num_classes)
        else:
            raise ValueError(f"Wrong fc-type input {self.fc_type}")
        return fc