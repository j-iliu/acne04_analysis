import os
import contextlib
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CocoDetection
from torchvision import transforms
import torch.distributed as dist
from classification_dataset import Acne04PatchDataset
from domain_transfer_dataset import DermnetAcneDataset
from torch.utils.data import Subset
from roboflow import Roboflow

def distributed_classification_dl(dermnet_dp, acne_version=2, batch_size=2, stage2=False, acne_ratio=0.5, jitter_on=True):

    train_ds, val_ds, test_ds = load_patches(stage2=stage2, acne_ratio=acne_ratio, jitter_on=jitter_on)

    dermnet_ds = DermnetAcneDataset(dermnet_dp)

    #train_ds = Subset(train_ds, range(100))
    #val_ds = Subset(val_ds, range(100))
    # test_ds = Subset(test_ds, range(4))
    # dermnet_ds = Subset(dermnet_ds, range(4))

    train_sampler = DistributedSampler(train_ds, shuffle=True,  drop_last=True)
    val_sampler   = DistributedSampler(val_ds,   shuffle=False, drop_last=False)

    cls_train_loader = DataLoader(
                    train_ds,
                    sampler=train_sampler,
                    batch_size=batch_size,
                    num_workers=2,
                    drop_last=True,
    )

    cls_val_loader = DataLoader(
                    val_ds,
                    sampler=val_sampler,
                    batch_size=batch_size,
                    num_workers=2,
    )

    cls_test_loader = DataLoader(
                    test_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=2,
    )

    dermnet_loader = DataLoader(
                    dermnet_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=2,
    )

    return cls_train_loader, cls_val_loader, cls_test_loader, dermnet_loader

def save_patches(version=2):
    
    all_saved = True
    paths_to_check = ["/kaggle/working/train_patches.pt", "/kaggle/working/val_patches.pt", "/kaggle/working/test_patches.pt"]

    for path in paths_to_check:
        if path is not None and os.path.exists(path):
            continue
        else:
            all_saved = False

    if all_saved:
        if not dist.is_initialized() or dist.get_rank() == 0:
            print("patches in cache")
        return
         
    acne_dp = load(version)
    train_data_path = f"{acne_dp}/train"
    train_data_ann  = f"{acne_dp}/train/_annotations.coco.json"

    val_data_path = f"{acne_dp}/valid"
    val_data_ann  = f"{acne_dp}/valid/_annotations.coco.json"

    test_data_path = f"{acne_dp}/test"
    test_data_ann  = f"{acne_dp}/test/_annotations.coco.json"

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            train_source = CocoDetection(root=train_data_path, annFile=train_data_ann, transform=transform)
            val_source   = CocoDetection(root=val_data_path,   annFile=val_data_ann,   transform=transform)
            test_source  = CocoDetection(root=test_data_path,  annFile=test_data_ann,  transform=transform)
    
    patch_input_train = "./train_patches"
    patch_input_val = "./val_patches"
    patch_input_test = "./test_patches"

    train_ds = Acne04PatchDataset(train_source, is_train=True, mosaic = True, patches_cache = "/kaggle/working/train_patches.pt", patch_input=patch_input_train)
    val_ds   = Acne04PatchDataset(val_source,   is_train=False, mosaic = True, patches_cache = "/kaggle/working/val_patches.pt", patch_input=patch_input_val)
    test_ds  = Acne04PatchDataset(test_source,  is_train=False, mosaic = True, patches_cache = "/kaggle/working/test_patches.pt", patch_input=patch_input_test)

def load_patches(stage2=False, acne_ratio=0.5, jitter_on=True):
    
    patch_input_train = "./train_patches"
    patch_input_val = "./val_patches"
    patch_input_test = "./test_patches"

    train_ds = Acne04PatchDataset(None, is_train=True, mosaic = True, patches_cache = "/kaggle/working/train_patches.pt", patch_input=patch_input_train, stage2=stage2, acne_ratio=acne_ratio, jitter_on=jitter_on)
    val_ds   = Acne04PatchDataset(None,   is_train=False, mosaic = True, patches_cache = "/kaggle/working/val_patches.pt", patch_input=patch_input_val, stage2=stage2, acne_ratio=acne_ratio, jitter_on=False)
    test_ds  = Acne04PatchDataset(None,  is_train=False, mosaic = True, patches_cache = "/kaggle/working/test_patches.pt", patch_input=patch_input_test,stage2=stage2, acne_ratio=acne_ratio, jitter_on=False)
    return train_ds, val_ds, test_ds
    
def load(version: int):
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace("jimmys-workspace-1ktw6").project("acne04-detection-i2hqg")
    dataset = project.version(version).download("coco")
    return dataset.location
