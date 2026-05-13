
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from IPython.display import clear_output
from tqdm import tqdm
from evaluation import calculate_accuracy
import torch.nn.functional as F


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
                 lr_steps: list = [10**3],
                 acc_every: int = 5,
                 preprocessor=None,
                 freeze_bn=False,
                 stage2=False,
                 acne_ratio=0.5
                 ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.model = DDP(self.model, device_ids=[gpu_id])
        self.name = name
        self.lr_scheduler = None
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.patience = patience
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.lr_steps = lr_steps
        self.acc_every = acc_every
        self.preprocessor = preprocessor
        self.freeze_bn = True
        self.stage2 = stage2
        self.acne_ratio = acne_ratio
        
        if stage2:
            self.name = name.replace('_best', '').replace('_last', '').strip('_')
        
        if optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(self.params, lr=lr, weight_decay=weight_decay)
        elif optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer == "RMSprop":
            self.optimizer = torch.optim.RMSprop(self.params, lr=lr, weight_decay=weight_decay, eps=1e-5)
        elif optimizer == "staggered AdamW":
                self.optimizer = torch.optim.AdamW([
                    {'params': self.model.module.layer4.parameters(), 'lr': self.lr},
                    {'params': self.model.module.fc.parameters(),     'lr': self.lr * 10},
                ], weight_decay=0.01)

        if self.stage2:
            #pos_weight = torch.tensor([0.1662, 1.8338]).to(gpu_id)
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.KLDivLoss(reduction='batchmean')
            
        #self.criterion = nn.KLDivLoss(reduction='batchmean')

    def run_batch(self, images, labels):
        self.optimizer.zero_grad()
        outputs = self.model(images)
        log_probs = None
        if self.stage2:
            loss = self.criterion(outputs, labels)
        else:
            log_probs = F.log_softmax(outputs, dim=1)
            loss = self.criterion(log_probs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()

    def run_epoch(self, epoch):
        total_loss = 0.0
        self.model.train()
        self.train_loader.sampler.set_epoch(epoch)

        for name, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if not self.stage2:
                    m.eval()
    
        loader = self.train_loader
        if self.gpu_id == 0:
            loader = tqdm(loader, desc=f"Epoch {epoch+1} [train]", leave=False)

        for images, labels in loader:
            images = images.to(self.gpu_id)
            labels = labels.to(self.gpu_id)

            images = self.preprocessor(images)
            total_loss += self.run_batch(images, labels)

        self.lr_scheduler.step()
        return total_loss / len(self.train_loader)

    def save_last(self):
        ckp = self.model.module.state_dict()
        save_name = f"{self.name}_last_stage_1.pt"
        if self.stage2:
            save_name = f"{self.name}_last_stage_2.pt"
        torch.save(ckp, save_name)
        print(f"{self.name} | Last saved")

    def save_best(self, epoch, val_loss):
        ckp = self.model.module.state_dict()
        save_name = f"{self.name}_best_stage_1.pt"
        if self.stage2:
            save_name = f"{self.name}_best_stage_2.pt"
        torch.save(ckp, save_name)
        print(f"Epoch {epoch} new best saved")

    def calculate_val_loss(self):
        self.model.eval()
        local_val_loss = 0.0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.gpu_id)
                labels = labels.to(self.gpu_id)
                images = self.preprocessor(images)
                outputs = self.model(images)
                log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
                if self.stage2:
                    loss = self.criterion(outputs, labels)
                else:
                    log_probs = F.log_softmax(outputs, dim=1)
                    loss = self.criterion(log_probs, labels)
                local_val_loss += loss.item()

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

            if self.acc_every > 0 and (epoch % self.acc_every == 0):
                calculate_accuracy(self.model, self.val_loader, self.gpu_id)

        if self.gpu_id == 0:
            print(f"Lowest val Loss: {lowest_val_loss} at epoch: {best_epoch}")
            print("Training complete")
            self.save_last()
            loss_plot = self.plot_losses(train_losses, val_losses)
            return loss_plot

        return None
