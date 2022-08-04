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
            'model': self.state_dict()}
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

            self.load_state_dict(data["model"])
            self.epoch = data['epoch']
            if optimizer is not None:
                optimizer.load_state_dict(data['optimizer'])
            return optimizer
