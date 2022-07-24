import torch

class Metric_tracker():
    def __init__(self, class_to_name, log_dir, set_k=None):
        if k is not None:
            if 1 not in set_k:
                set_k.append(1)
            set_k.sort()
            self.set_k = set_k
        else:
            self.set_k = [1,3,5,10]
        
        self.topk_tp = {}
        self.top1_fp = torch.zeros(len(class_to_name), torch.int16)
        self.samples_per_class = torch.zeros(len(class_to_name), torch.int16)
        for k in self.set_k:
            self.topk_tp[k] = torch.zeros(len(class_to_name), torch.int16)
        self.losses = []

    def update_topk_TruePositives(self, y_true, y_pred):
        predicted_classes = torch.argsort(y_pred, axis=-1, descending=True)
        for k in self.set_k:
            for gt, pred in zip(y_true, predicted_classes):
                if k == 1: #At first iteration.
                    self.num_samples[gt.item()]+=1 #counting class item.
                    top_k = pred[:k]
                    self.topk_tp[k][gt.item()]+=torch.sum(gt == top_k).item()
                    self.top1_fp+= torch.sum(gt != top_k).item()
                else:
                    top_k = pred[:k]
                    self.topk_tp[k][gt.item()]+=torch.sum(gt == top_k).item()

    def update(self, loss, y_true, y_pred):
        self.update_topk_TruePositives(y_true, y_pred)
        self.losses.append(loss)

    def result(self):
        samples_per_class, zero_classes = self.cal_samples()
        precisions = self.cal_precision(samples_per_class)
        recalls = self.cal_recall(samples_per_class)
        topk_acc = self.cal_topk_accuracy(samples_per_class) 
        epoch_loss = torch.mean(self.losses)
        return samples_per_class, zero_classes, precisions, recalls, topk_acc, epoch_loss

    def call_samples(self):
        #If there is no samples in a class, record it at zero_classes
        #num_samples should be larger than 0 to avoid divided by zero when you calculate metrics.
        samples_per_class = self.samples_per_class
        zero_classes = torch.where(samples_per_class==0)[0] #Detect no sample classes.
        samples_per_class = torch.where(samples_per_class==0, 1, samples_per_class)
        return samples_per_class, zero_classes


    def cal_precision(self, samples_per_class):
        precisions = torch.div(self.topk_tp[1], self.topk_tp[1]+self.top1_fp)

    def cal_recall(self, samples_per_class):
        recall = torch.div(self.topk_tp[1], samples_per_class)

    def cal_topk_accuracy(self, samples_per_class):
        topk_acc = {}
        for k in self.set_k:
            topk_acc[k] = torch.div( torch.sum(self.topk_tp[k]), torch.sum(samples_per_class))
        return topk_acc
