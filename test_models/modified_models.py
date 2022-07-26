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
                data = torch.load(path_to_pt, map_location=next(self.parameters()).device)
            else:
                data = torch.load(path_to_pt, map_location=lambda storage, loc: storage)

            self.load_state_dict(data["model"])
            self.epoch = data['epoch']
            if optimizer is not None:
                optimizer.load_state_dict(data['optimizer'])
            return optimizer


class HierarchicalClassifier(nn.Module):
    def __init__(self, loss_fn, baseline, num_classes):
        super(HierarchicalClassifier, self).__init__()
        self.loss_fn = loss_fn
        self.baseline = baseline

        self.lvl1_classifier = nn.Linear(1792, num_classes[0])
        self.channel_attention = nn.Linear(num_classes[0], 480)
        self.sigmoid = nn.Sigmoid()
        self.linear_lvl2 = nn.Linear(1792, 480)
        self.lvl2_classifier = nn.Linear(480+num_classes[0], num_classes[1])
        
    def forward(self, images):
        x = self.baseline.embedding(self.baseline.patch_embedding(images))
        
        #x=>level1=>parent info
        #1792 => 960 => non-linear => lvl1_classifier
        
        #            => 480 => 
        #     => 960 => 480 => concat => 960 => non-linear => lvl2_classifier
        
        level_1 = self.lvl1_classifier(x)
        
        attention_w = self.sigmoid(self.channel_attention(level_1))       
        lvl2_rep = torch.cat((level_1, attention_w*self.linear_lvl2(x)), dim=-1)
        level_2 = self.lvl2_classifier(lvl2_rep) #attetion weight*level2 rep
        return level_1, level_2
    
    def predict(self, images):
        _, level2 = self(images)
        return nn.Softmax(dim=-1)(level2)

    def train_step(self, images, y_true, optimizer, lr_scheduler=None):
        images = images.to(next(self.parameters()).device)
        y1_true, y2_true = y_true
        y1_true = y1_true.to(next(self.parameters()).device)
        y2_true = y2_true.to(next(self.parameters()).device)

        optimizer.zero_grad()
        level_1, level_2 = self(images) #self.fc(self.embedding(images))
        loss = self.loss_fn([level_1, level_2], [y1_true, y2_true])
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        return loss, [level_1, level_2]

    def test_step(self, images, y_true):
        images = images.to(next(self.parameters()).device)
        y1_true, y2_true = y_true
        y1_true = y1_true.to(next(self.parameters()).device)
        y2_true = y2_true.to(next(self.parameters()).device)

        with torch.no_grad():
            level_1, level_2 = self(images) #self.fc(self.embedding(images))
            loss = self.loss_fn([level_1, level_2], [y1_true, y2_true])
        return loss, [level_1, level_2]

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
                data = torch.load(path_to_pt, map_location=next(self.parameters()).device)
            else:
                data = torch.load(path_to_pt, map_location=lambda storage, loc: storage)

            self.load_state_dict(data["model"])
            self.epoch = data['epoch']
            if optimizer is not None:
                optimizer.load_state_dict(data['optimizer'])
            return optimizer


class HierarchicalClassifier_v2(HierarchicalClassifier):
    def __init__(self, loss_fn, baseline, num_classes):
        super(HierarchicalClassifier_v2, self).__init__(loss_fn, baseline, num_classes)
        self.channel_attention = nn.Linear(num_classes[0], 1792)
        self.lvl2_classifier = nn.Linear(1792+num_classes[0], num_classes[1])
        
    def forward(self, images):
        x = self.baseline.embedding(self.baseline.patch_embedding(images))
               
        level_1 = self.lvl1_classifier(x)
        
        attention_w = self.sigmoid(self.channel_attention(level_1))       
        lvl2_rep = torch.cat((level_1, attention_w*x), dim=-1)
        level_2 = self.lvl2_classifier(lvl2_rep) #attetion weight*level2 rep
        return level_1, level_2
    
class HierarchicalClassifier_v3(HierarchicalClassifier):
    def __init__(self, loss_fn, baseline, num_classes):
        super(HierarchicalClassifier_v3, self).__init__(loss_fn, baseline, num_classes)
        #self.channel_attention = nn.Linear(num_classes[0], 1792)
        self.lvl2_classifier = nn.Linear(1792, num_classes[1])
        
    def forward(self, images):
        x = self.baseline.embedding(self.baseline.patch_embedding(images))
               
        level_1 = self.lvl1_classifier(x)
        level_2 = self.lvl2_classifier(x)
        return level_1, level_2
    
class HierarchicalClassifier_v4(HierarchicalClassifier):
    def __init__(self, loss_fn, baseline, num_classes):
        super(HierarchicalClassifier_v4, self).__init__(loss_fn, baseline, num_classes)
        self.lvl2_classifier = nn.Linear(1792+num_classes[0], num_classes[1])
        
    def forward(self, images):
        x = self.baseline.embedding(self.baseline.patch_embedding(images))
               
        level_1 = self.lvl1_classifier(x)
        #attention_w = self.sigmoid(self.channel_attention(level_1))       
        lvl2_rep = torch.cat((level_1, x), dim=-1)
        level_2 = self.lvl2_classifier(lvl2_rep) #attetion weight*level2 rep
        return level_1, level_2