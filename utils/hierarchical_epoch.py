from tqdm import tqdm
from utils import metrics

def train_epoch(model, optimizer, train_dataloader, val_dataloader, metrics, epoch, lr_scheduler=None):
    model.train()
    metrics["train"].reset()
    t = tqdm(train_dataloader)
    for batch in t:
        images, y1_true, y2_true = batch
        loss, y_pred = model.train_step(images, y1_true, y2_true, optimizer, lr_scheduler=lr_scheduler)
        metrics['train'].update(loss, y2_true, y_pred)
        t.set_description(f"epoch : {epoch} train-step loss : {metrics['train'].losses[-1]:.3f}")
    
    model.eval()
    metrics["val"].reset()
    t = tqdm(val_dataloader)
    for batch in t:
        images, y1_true, y2_true = batch
        loss, y_pred = model.test_step(images, y1_true, y2_true)
        metrics["val"].update(loss, y2_true, y_pred)
        t.set_description(f"epoch : {epoch} val-step loss : {metrics['val'].losses[-1]:.3f}")

def test_epoch(model, test_dataloader, metric):
    model.eval()
    metric.reset()
    t = tqdm(test_dataloader)
    for batch in t:
        images, y1_true, y2_true = batch
        loss, y_pred = model.test_step(images, y1_true, y2_true)
        metric.update(loss, y2_true, y_pred)
        t.set_description(f"test-step loss : {metric.losses[-1]:.3f}")