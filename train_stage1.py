
import os
import sys
import torch
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import json
from IPython.display import clear_output
import argparse
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from evaluation import calculate_accuracy
from classification_loaders import load, distributed_classification_dl
from classification_model_trainer import Trainer, ddp_setup
from load_models import load_vggface2, load_resnet50
import roboflow as Roboflow
from classification_loaders import save_patches 
from histogram_matching import build_histogram_preprocessor

def parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--name", type=str, default="model")
    parser.add_argument("--steps", type=int, nargs='+', default=[1000])
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--use_histogram_matching", action='store_true', default=False)
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()
    

def main(rank, world_size, args):
    ddp_setup(rank, world_size)
    dermnet_dp = "/kaggle/input/datasets/shubhamgoel27/dermnet/test"
    try:
        model = None
        preprocessor = None
        if "vggface2" in args.name:
            model, preprocessor = load_vggface2()
            if rank == 0:
                print("using resnet50 pretrained on vggface2")
        else:
            model, preprocessor = load_resnet50()

        if args.use_histogram_matching:
            preprocessor = build_histogram_preprocessor(
                dermnet_root="/kaggle/input/datasets/shubhamgoel27/dermnet/train",
                base_preprocessor=preprocessor,
                device=f"cuda:{rank}",
                rank=rank,
                visualize=True,
                save_path="/kaggle/working/style_images.png"
            )
        
        dist.barrier()
        train_dl, val_dl, test_dl = distributed_classification_dl(batch_size=args.batch_size, stage2=False, jitter_on = not args.use_histogram_matching)
        trainer = Trainer(
            model=model,
            preprocessor=preprocessor,
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            train_loader=train_dl,
            val_loader=val_dl,
            optimizer=args.optimizer,
            gpu_id=rank,
            save_every=10,
            name=args.name,
            lr_steps=args.steps,
            patience=args.patience,
            acc_every=-1,
        )
        
        losses_plot = trainer.train(args.epochs)

        if rank == 0:
            losses_plot.savefig(f"{args.name}_loss_curves.png")
            
    finally:
        destroy_process_group()
    
if __name__ == "__main__":
    args = parse() 
    save_patches(version=2)
    load(version=2)
    world_size = torch.cuda.device_count()    

    mp.spawn(main, args=(world_size, args), nprocs=world_size)
