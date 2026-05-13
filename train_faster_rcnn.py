
import os
import sys
import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.distributed as dist
from faster_rcnn_dataset import load, distributed_fast_rcnn_dl
from trainer import Trainer, ddp_setup
import json
from IPython.display import clear_output
import argparse
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from evaluation import calculate_map

def load_model():
    fasterrcnn = fasterrcnn_resnet50_fpn(weights="DEFAULT", trainable_backbone_layers=3)
    in_features = fasterrcnn.roi_heads.box_predictor.cls_score.in_features
    fasterrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    fasterrcnn.rpn.anchor_generator = anchor_generator
    
    return fasterrcnn

def parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--name", type=str, default="model")
    parser.add_argument("--steps", type=int, nargs='+', default=[1000])
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--map_every", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=4)
    
    return parser.parse_args()
    

def main(rank, world_size, args):
    ddp_setup(rank, world_size)
    try:
        model = load_model()
        if rank == 0:
            datapath = load(version=6)
        dist.barrier()
        if rank != 0:
            datapath = load(version=6)
        if rank == 0:
            clear_output(wait=True)
        dist.barrier()
        train_dl, val_dl, test_dl = distributed_fast_rcnn_dl(datapath, batch_size=args.batch_size)
        trainer = Trainer(
            model=model,
            lr=args.lr,
            momentum=0.9,
            weight_decay=0.0005,
            train_loader=train_dl,
            val_loader=val_dl,
            optimizer=args.optimizer,
            gpu_id=rank,
            save_every=10,
            name=args.name,
            lr_steps=args.steps,
            patience=args.patience,
            map_every=args.map_every,
        )
        
        losses_plot, _ = trainer.train(args.epochs)

        if rank == 0:
            print("Final MAP")
        results, pr_plot = calculate_map(model, test_dl, rank, IoU=True)

        if rank == 0:
            losses_plot.savefig(f"{args.name}_loss_curves.png")
            pr_plot.savefig(f"{args.name}_pr_curve.png")
            
    finally:
        destroy_process_group()
    
if __name__ == "__main__":
   args = parse()
   world_size = torch.cuda.device_count()    
   mp.spawn(main, args=(world_size, args), nprocs=world_size)
