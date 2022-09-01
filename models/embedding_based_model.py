from turtle import forward
import torch
from torch import nn
from embeddings.set_distance import SetDist

class mahalanobis_classififer():
    def __init__(self, base_model, stats):
        self.base_model = base_model
        self.dist = SetDist()
        self.stats = stats
    def __call__(self, x):
        y = self.base_model(x)


class Mahalanobis(nn.Module):
    def __init__(self, stats):
        super().__init__()
        self.stats = stats
    def mahalanobis_distance(self, embedding, stats):
        mean, inv_covariance = stats
        delta = (embedding - mean)
        distance = (torch.matmul(delta, inv_covariance) * delta).sum()
        distance = torch.sqrt(distance)
        return distance
    def forward(self, x):
        dist = self.mahalanobis_distance(x, self.stats[0])
        return dist

