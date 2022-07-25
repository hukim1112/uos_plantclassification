from tqdm import tqdm
from utils import metrics

def train_epoch(model, optimizer, train_dataloader, val_dataloader, metrics, lr_scheduler=None):
    model.train()
    metrics["train"].reset()
    t = tqdm(train_dataloader)
    for batch in t:
        images, y_true = batch
        loss, y_pred = model.train_step(images, y_true, optimizer, lr_scheduler=lr_scheduler)
        metrics['train'].update(loss, y_true, y_pred)
        t.set_description(f"train-step loss : {metrics['train'].losses[-1]:.3f}")
    
    model.eval()
    metrics["val"].reset()
    t = tqdm(val_dataloader)
    for batch in t:
        images, y_true = batch
        loss, y_pred = model.test_step(images, y_true)
        metrics["val"].update(loss, y_true, y_pred)
        t.set_description(f"val-step loss : {metrics['val'].losses[-1]:.3f}")

def test_epoch(model, test_dataloader, metrics):
    model.eval()
    metrics["test"].reset()
    t = tqdm(test_dataloader)
    for batch in t:
        images, y_true = batch
        loss, y_pred = model.test_step(images, y_true)
        metrics["test"].update(loss, y_true, y_pred)
        t.set_description(f"test-step loss : {metrics['test'].losses[-1]:.3f}")