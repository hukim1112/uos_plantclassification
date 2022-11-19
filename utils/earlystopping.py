import torch
import numpy as np

class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=True, delta=0, is_loss=True, path='checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.is_loss = is_loss
        self.path = path
        if is_loss:
            self.previous_best = np.Inf
        else:
            self.previous_best = 0.0


    def __call__(self, metric, epoch, model, optimizer):
        if self.is_loss:
            score = -metric
            delta = -self.delta
        else:
            score = metric
            delta = self.delta


        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(metric, epoch, model, optimizer)
        elif score+delta < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(metric, epoch, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, metric, epoch, model, optimizer, minimum=True):
        '''validation loss가 감소하거나(검증정확도의 경우 증가) 모델을 저장한다.'''
        if self.verbose:
            print(f'Metric improved ({self.previous_best:.6f} --> {metric:.6f}).  Saving model ...')
        model.save(epoch, self.path, optimizer=optimizer)
        self.previous_best = metric