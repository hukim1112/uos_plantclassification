import os
import torch
from torch import nn
import timm

class Baseline(nn.Module):
    def __init__(self, loss_fn):
        super(Baseline, self).__init__()
        self.loss_fn = loss_fn
        self.embedding = None
        self.fc = None
        self.epoch = None       

    def forward(self, images):
        return self.fc(self.embedding(images))

    def predict(self, images):
        return nn.Softmax(dim=-1)(self(images))

    def train_step(self, images, y_true, optimizer, lr_scheduler=None):
        images = images.to(next(self.parameters()).device)
        y_true = y_true.to(next(self.parameters()).device)

        self.optimizer.zero_grad()
        y_pred = self(images) #self.fc(self.embedding(images))
        loss = self.loss_fn(y_true, y_pred)
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        return loss, y_pred

    def test_step(self, images, y_true):
        with torch.no_grad():
            y_pred = self(images) #self.fc(self.embedding(images))
            loss = self.loss_fn(y_true, y_pred)
        return loss, y_pred

    def save(self, epoch, weight_path):
        dir = os.path.dirname(weight_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        d = {'epoch': epoch,
            'embedding': self.embedding.state_dict(),
            'fc' : self.fc.state_dict(),
            'optimizer': self.optimizer.state_dict()}
        torch.save(d, weight_path) 

    def load(self, weight_path):
        if not os.path.exists(weight_path):
            print('Loading {weight_path} : error')
        else:
            if torch.cuda.is_available():
                data = torch.load(weight_path)
            else:
                data = torch.load(weight_path, map_location=lambda storage, loc: storage)

            self.embedding.load_state_dict(data["embedding"])
            self.fc.load_state_dict(data['fc'])
            self.optimizer.load_state_dict(data['optimizer'])
            self.epoch = data['epoch']


class EfficientB0(Baseline):
    def __init__(self, num_classes, loss_fn, fc_type="deep"):
        super(EfficientB0, self).__init__(loss_fn)
        self.num_classes = num_classes
        self.fc_type = fc_type
        self.embedding = self.get_backbone()
        self.fc = self.get_fc()     

    def get_backbone(self):
        cnn = timm.create_model('efficientnet_b0', pretrained=False)
        return nn.Sequential( *list(cnn.children())[:-1])
    
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
