
import torch
import torch.nn.functional as F  # Missing for softmax
import torch.distributed as dist
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.classification import (
    BinaryAccuracy, 
    BinaryAUROC, 
    BinaryF1Score, 
    BinaryAveragePrecision as BinaryAUPRC
)

def calculate_map(model, data_loader, device, IoU = False):
    model.to(device)
    model.eval()

    metric = MeanAveragePrecision(box_format='xyxy', extended_summary=True, sync_on_compute=True).to(device)

    pred = []
    gt = []

    is_rank_zero = not dist.is_initialized() or dist.get_rank() == 0
    
    with torch.no_grad():
        for imgs, tgts in tqdm(data_loader, desc="Calculating Map", disable=not is_rank_zero, leave=False):
            imgs = [img.to(device) for img in imgs]
            tgts = [{k: v.to(device) for k, v in t.items()} for t in tgts]

            outputs = model(imgs)
            metric.update(outputs, tgts)

    dist.barrier()
    result = metric.compute()

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"mAP@50: {result['map_50']:.4f} | mAP@50-95: {result['map']:.4f}| mAP@50 (small / medium / large): {result['map_small']:.4f} / {result['map_medium']:.4f} / {result['map_large']:.4f}")
    
    if IoU:
        precision = result['precision'][0, :, 0, 0, -1].cpu().numpy()
        recall_y = np.linspace(0, 1, 101)

        fig = plt.figure(figsize=(8, 6))
        plt.plot(recall_y, precision)
        plt.title('Precision Recall Curve @ T=0.5')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])

        return result, fig
        
    return result, None

def calculate_accuracy(model, data_loader, device, preprocessor=None):
    model.eval()

    is_rank_zero = not dist.is_initialized() or dist.get_rank() == 0

    accuracy_metric = BinaryAccuracy().to(device)
    auroc_metric    = BinaryAUROC().to(device)
    f1_metric       = BinaryF1Score().to(device)
    auprc_metric    = BinaryAUPRC().to(device)

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating", disable=not is_rank_zero, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            if preprocessor is not None:
                images = preprocessor(images)

            if labels.ndim == 2:
                labels = labels.argmax(dim=1)
            labels = labels.long()

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)

            accuracy_metric.update(preds, labels)
            auroc_metric.update(probs, labels)
            auprc_metric.update(probs, labels)
            f1_metric.update(preds, labels)

    accuracy = accuracy_metric.compute().item()
    auroc    = auroc_metric.compute().item()
    f1       = f1_metric.compute().item()
    auprc    = auprc_metric.compute().item()

    if is_rank_zero:
        print(f"Accuracy: {accuracy:.4f} | AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | F1: {f1:.4f}")

    return accuracy, auroc, f1, auprc
    
