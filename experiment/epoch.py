from tqdm import tqdm
from utils import metrics

def train_epoch(model, optimizer, train_dataloader, val_dataloader, metrics, lr_scheduler=None):
    model.train()
    metrics["train"].reset()
    for batch in tqdm(train_dataloader, desc="train-steps"):
        images, y_true = batch
        loss, y_pred = model.train_step(images, y_true, optimizer, lr_scheduler=lr_scheduler)
        metrics['train'].update(loss, y_true, y_pred)
    model.eval()
    metrics["val"].reset()
    for batch in tqdm(val_dataloader, desc='val-steps'):
        images, y_true = batch
        loss, y_pred = model.test_step(images, y_true)
        metrics["val"].update(loss, y_true, y_pred)

def test_epoch(model, test_dataloader, metrics):
    model.eval()
    metrics["test"].reset()
    for batch in tqdm(test_dataloader, desc="test-steps"):
        images, y_true = batch
        loss, y_pred = model.test_step(images, y_true)
        metrics["test"].update(loss, y_true, y_pred)