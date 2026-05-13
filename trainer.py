
import os
import torch
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from IPython.display import clear_output
from tqdm import tqdm
from evaluation import calculate_map

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=torch.device(f"cuda:{rank}"))

class Trainer:
    def __init__(self, 
                model: torch.nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                optimizer: str,
                gpu_id: int,
                save_every: int,
                name: str = "model",
                lr: float = 0.005,
                weight_decay: float = 0.0001,
                momentum: float = 0.9,
                patience: int = 6,
                lr_steps: int = [10**3],
                map_every: int = 5,
               ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gpu_id = gpu_id
        self.save_every = save_every
        self.model = model.to(gpu_id)
        print(f"[rank {self.gpu_id}] after DDP wrap", flush=True)    
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.name = name
        self.lr_scheduler = None
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.patience = patience
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.lr_steps = lr_steps
        self.map_every = map_every
        
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.params, lr=lr, weight_decay=weight_decay, eps=1e-5)


    def run_batch(self, images, targets):
        self.optimizer.zero_grad()
        loss_dict = self.model(images,targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        self.optimizer.step()
        return losses.item()
        
    def run_epoch(self, epoch):
        total_loss = 0
        self.model.train()
        self.train_loader.sampler.set_epoch(epoch)

        loader = self.train_loader

        if self.gpu_id == 0:
            loader = tqdm(loader, desc=f"Epoch {epoch+1} [train]", leave=False)
        
        for images, targets in loader:
            images = list(img.to(self.gpu_id) for img in images)
            targets = [{k: v.to(self.gpu_id) for k, v in t.items()} for t in targets]
            total_loss += self.run_batch(images, targets)
        
        self.lr_scheduler.step()
        loss_tensor = torch.tensor(total_loss).to(self.gpu_id)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        return (loss_tensor / (len(self.train_loader) * dist.get_world_size())).item()
            
    def save_last(self):
        ckp = self.model.module.state_dict()
        torch.save(ckp, f"{self.name}_last.pt")
        print(f"{self.name} | Last saved")

    def save_best(self, epoch, val_loss):
        best = self.model.module.state_dict()
        torch.save(best, f"{self.name}_best.pt")
        #print(f"{self.name} | Epoch {epoch} | New best saved | Validation Loss {val_loss}")
        print(f"Epoch {epoch} new best saved")

    def calculate_val_loss(self):
        self.model.train()
        local_val_loss = 0.0
        with torch.no_grad():        
            for images, targets in self.val_loader:
                images = list(image.to(self.gpu_id) for image in images)
                targets = [{k: (v.to(self.gpu_id) if isinstance(v, torch.Tensor) else v)
                            for k, v in t.items()} for t in targets]
                      
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                local_val_loss += losses.item()
                
        val_loss_tensor = torch.tensor(local_val_loss).to(self.gpu_id)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        global_val_loss = val_loss_tensor / (len(self.val_loader) * dist.get_world_size())
        
        return global_val_loss.item()
        
    def plot_losses(self, train_losses, val_losses):
        clear_output(wait=True)
        fig = plt.figure(figsize=(8, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        return fig

    def train(self, epochs: int):
        #self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.lr_floor_epoch,eta_min=0.00001)
        
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.lr_steps, gamma=0.1)
        
        train_losses = []
        val_losses = []

        lowest_val_loss = float('inf')
        best_epoch = 0
        
        for epoch in range(epochs):

            train_loss = self.run_epoch(epoch)
            val_loss = self.calculate_val_loss()     

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < lowest_val_loss:
                best_epoch = epoch
                lowest_val_loss = val_loss
                if self.gpu_id == 0:
                    self.save_best(epoch, val_loss)
            
            if self.gpu_id == 0:
                print(f"Epoch {epoch+1}/{epochs}, train Loss: {train_loss}, val Loss: {val_loss}")

            if epoch - best_epoch >= self.patience:
                if self.gpu_id == 0:
                    print(f"Early stopping at epoch {epoch+1} (no improvement for {self.patience} epochs)")
                break

            if self.map_every > 0 and (epoch % self.map_every == 0):
                calculate_map( self.model, self.val_loader, self.gpu_id, IoU=False)

        result, pr_plot = calculate_map(self.model, self.val_loader, self.gpu_id, IoU=True)

        if self.gpu_id  == 0:
            print(f"Lowest val Loss: {lowest_val_loss} at epoch: {best_epoch}" )
            print("Training complete")
            self.save_last()
            loss_plot = self.plot_losses(train_losses, val_losses)
            
            return loss_plot, pr_plot

        return None, None

    
