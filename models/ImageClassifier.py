import os
import torch
from torch import nn
import timm

class Baseline(nn.Module):
    def __init__(self, loss_fn):
        super(Baseline, self).__init__()
        self.loss_fn = loss_fn
        self.patch_embedding = None
        self.embedding = None
        self.fc = None

    def forward(self, images):
        return self.fc(self.embedding(self.patch_embedding(images)))

    def predict(self, images):
        return nn.Softmax(dim=-1)(self(images))

    def train_step(self, images, y_true, optimizer, lr_scheduler=None):
        images = images.to(next(self.parameters()).device)
        y_true = y_true.to(next(self.parameters()).device)

        optimizer.zero_grad()
        y_pred = self(images) #self.fc(self.embedding(images))
        loss = self.loss_fn(y_pred, y_true)
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        return loss, y_pred

    def test_step(self, images, y_true):
        images = images.to(next(self.parameters()).device)
        y_true = y_true.to(next(self.parameters()).device)

        with torch.no_grad():
            y_pred = self(images) #self.fc(self.embedding(images))
            loss = self.loss_fn(y_pred, y_true)
        return loss, y_pred

    def save(self, epoch, path_to_pt, optimizer=None):
        dir = os.path.dirname(path_to_pt)
        if not os.path.exists(dir):
            os.makedirs(dir)
        d = {'epoch': epoch,
            'patch_embedding': self.patch_embedding.state_dict(),
            'fc' : self.fc.state_dict()}
        if optimizer is not None:
            d['optimizer'] = optimizer.state_dict()
        torch.save(d, path_to_pt) 

    def load(self, path_to_pt, optimizer=None):
        if not os.path.exists(path_to_pt):
            print('Loading {weight_path} : error')
        else:
            if torch.cuda.is_available():
                data = torch.load(path_to_pt)
            else:
                data = torch.load(path_to_pt, map_location=lambda storage, loc: storage)

            self.patch_embedding.load_state_dict(data["patch_embedding"])
            self.fc.load_state_dict(data['fc'])
            self.epoch = data['epoch']
            if optimizer is not None:
                optimizer.load_state_dict(data['optimizer'])
            return optimizer


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
