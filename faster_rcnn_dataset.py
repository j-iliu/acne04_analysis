
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torchvision.transforms import v2 as T 
from roboflow import Roboflow
import torch.distributed as dist
from torch.utils.data import Subset
import sys
import contextlib 

class FasterRCNNDataset(torch.utils.data.Dataset):
    
    def __init__(self, base, is_train=False):
        self.base = base
        self.is_train = is_train
        self.jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

    def __len__(self):
        return len(self.base)

    

    def __getitem__(self, idx):
        img, target = self.base[idx]
    
        _, H, W = img.shape
        scale = min(800 / min(H, W), 1333 / max(H, W))
        new_H, new_W = int(round(H * scale)), int(round(W * scale))
        img = T.functional.resize(img, [new_H, new_W],
                                  interpolation=T.InterpolationMode.BILINEAR,
                                  antialias=True)
    
        boxes = []
        labels = []
    
        is_flipped = False
        if self.is_train:
            img = self.jitter(img)
        if self.is_train and torch.rand(1) > 0.5:
            img = T.functional.hflip(img)
            is_flipped = True
        for ann in target:
            x, y, w, h = ann['bbox']
            x, y, w, h = x * scale, y * scale, w * scale, h * scale
            if is_flipped:
                x = img.shape[2] - x - w
            boxes.append([x, y, x+w, y+h])
            labels.append(1)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        return img, {"boxes": boxes, "labels": labels}

def collate_fn(batch):
    return tuple(zip(*batch))

def load(version: int):
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
            project = rf.workspace("jimmys-workspace-1ktw6").project("acne04-detection-i2hqg")
            dataset = project.version(version).download("coco")
    
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("Loaded dataset succesfully")
    return dataset.location

def distributed_fast_rcnn_dl(dp, batch_size=2):
       
    train_data_path = f"{dp}/train"
    train_data_ann  = f"{dp}/train/_annotations.coco.json"
    
    val_data_path   = f"{dp}/valid"
    val_data_ann    = f"{dp}/valid/_annotations.coco.json"
    
    test_data_path  = f"{dp}/test"
    test_data_ann   = f"{dp}/test/_annotations.coco.json"

    transform = transforms.Compose([
            transforms.ToTensor(),
    ])
    
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            train_dataset = CocoDetection(root=train_data_path, annFile=train_data_ann, transform=transform)
            val_dataset = CocoDetection(root=val_data_path, annFile=val_data_ann, transform=transform)
            test_dataset = CocoDetection(root=test_data_path, annFile=test_data_ann, transform=transform)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print("loaded annotations into memory")

    train_dataset = Subset(train_dataset, range(700))
    val_dataset = Subset(val_dataset, range(15))
    
    train_ds = FasterRCNNDataset(train_dataset, is_train=True)
    val_ds = FasterRCNNDataset(val_dataset, is_train=False)
    test_ds = FasterRCNNDataset(test_dataset, is_train=False)

    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    val_sampler   = DistributedSampler(val_ds,   shuffle=False, drop_last=False)
    
    fcnn_train_loader = DataLoader(
                    train_ds, 
                    sampler=train_sampler, 
                    batch_size=batch_size, 
                    num_workers=2, 
                    collate_fn=collate_fn,
                    drop_last=True,
    )
    
    fcnn_val_loader = DataLoader(
                    val_ds, 
                    sampler=val_sampler,
                    batch_size=batch_size, 
                    shuffle=False, 
                    num_workers=2, 
                    collate_fn=collate_fn)
    
    fcnn_test_loader = DataLoader(
                    test_ds, 
                    batch_size=batch_size, 
                    shuffle=False, 
                    num_workers=2, 
                    collate_fn=collate_fn)

    return fcnn_train_loader, fcnn_val_loader, fcnn_test_loader
