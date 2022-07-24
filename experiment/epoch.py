import tqdm
from utils import metrics

def train_epoch(model, optimizer, train_dataloader, val_dataloader, metrics, lr_scheduler=None):
    model.train()
    for batch in tqdm(train_dataloader, desk="train steps"):
        images, y_true = batch
        loss, y_pred = model.train_step(images, y_true, optimizer, lr_scheduler=lr_scheduler)
        metrics['train'].update(y_true, y_pred)
        metrics['train'].loss(loss)
    model.eval()
    for batch in tqdm(val_dataloader, desk='val steps'):
        images, y_true = batch
        loss, y_pred = model.test_step(images, y_true)
        metrics['val'].update(y_true, y_pred)
        metrics['val'].loss(loss)