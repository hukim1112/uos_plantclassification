import torch
import pandas as pd
from os.path import join

class Metric_tracker():
    def __init__(self, split, class_to_name, log_dir, set_k=None):
        self.split = split
        self.class_to_name = class_to_name
        self.log_dir = log_dir
        self.set_k = set_k
        if self.set_k is not None:
            if 1 not in self.set_k:
                self.set_k.append(1)
            self.set_k.sort()
        else:
            self.set_k = [1,3,5,10]
        self.reset()

    def reset(self):
        self.topk_tp = {}
        self.top1_fp = torch.zeros(len(self.class_to_name), dtype=torch.int)
        self.samples_per_class = torch.zeros(len(self.class_to_name), dtype=torch.int)
        for k in self.set_k:
            self.topk_tp[k] = torch.zeros(len(self.class_to_name), dtype=torch.int)
        self.losses = []

    def update_topk_TruePositives(self, y_true, y_pred):
        predicted_classes = torch.argsort(y_pred, axis=-1, descending=True)
        for k in self.set_k:
            for gt, pred in zip(y_true, predicted_classes):
                if k == 1: #At first iteration.
                    self.samples_per_class[gt.item()]+=1 #counting class item.
                    top_k = pred[:k]
                    self.topk_tp[k][gt.item()]+=torch.sum(gt == top_k).item()
                    self.top1_fp[k]+= torch.sum(gt != top_k).item()
                else:
                    top_k = pred[:k]
                    self.topk_tp[k][gt.item()]+=torch.sum(gt == top_k).item()

    def update(self, loss, y_true, y_pred):
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()
        self.update_topk_TruePositives(y_true, y_pred)
        self.losses.append(loss)

    def result(self):
        samples_per_class, zero_classes = self.cal_samples()
        precisions = self.cal_precision(samples_per_class)
        recalls = self.cal_recall(samples_per_class)
        topk_acc = self.cal_topk_accuracy(samples_per_class) 
        epoch_loss = self.cal_epoch_loss()
        return samples_per_class, zero_classes, precisions, recalls, topk_acc, epoch_loss

    def to_writer(self, writer, epoch, lr=None):
        samples_per_class, zero_classes, precisions, recalls, topk_acc, epoch_loss = self.result()
        writer.add_scalar(f"Loss/{self.split}", epoch_loss.item(), epoch)
        writer.add_scalar(f"acc/{self.split}", topk_acc[1].item(), epoch)
        writer.add_scalar(f"balanced_acc/{self.split}", torch.mean(recalls).item(), epoch)
        for k in self.set_k:
            writer.add_scalar(f"top-{k} acc/{self.split}", topk_acc[k].item(), epoch)
        if lr is not None:
            writer.add_scalar(f"learning_rate/{self.split}", lr, epoch)

    def to_csv(self, path):
        samples_per_class, zero_classes, precisions, recalls, topk_acc, epoch_loss = self.result()
        df1 = pd.DataFrame({"name" : list(self.class_to_name.values()), 
                           "samples_per_class" : samples_per_class.numpy(),
                           "precision" : precisions.numpy(),
                           "recall" : recalls.numpy()})
        df2 = pd.DataFrame({"name" : [k for k in self.set_k]+["balanced_acc", "loss"], 
                           "metric" : [topk_acc[k] for k in self.set_k]+[torch.mean(recalls).item(), epoch_loss.item()]})
        df1.to_csv(join(path, f"categorical_metrics_{self.split}.csv"))
        df2.to_csv(join(path, f"overall_metric_{self.split}.csv"))


    def cal_samples(self):
        #If there is no samples in a class, record it at zero_classes
        #num_samples should be larger than 0 to avoid divided by zero when you calculate metrics.
        samples_per_class = self.samples_per_class
        zero_classes = torch.where(samples_per_class==0)[0] #Detect no sample classes.
        samples_per_class = torch.where(samples_per_class==0, torch.tensor(1,dtype=torch.int), samples_per_class)
        return samples_per_class, zero_classes

    def cal_precision(self, samples_per_class):
        return torch.div(self.topk_tp[1], self.topk_tp[1]+self.top1_fp)

    def cal_recall(self, samples_per_class):
        return torch.div(self.topk_tp[1], samples_per_class)

    def cal_topk_accuracy(self, samples_per_class):
        topk_acc = {}
        for k in self.set_k:
            topk_acc[k] = torch.div( torch.sum(self.topk_tp[k]), torch.sum(samples_per_class))
        return topk_acc

    def cal_epoch_loss(self):
        return torch.mean(torch.stack(self.losses))