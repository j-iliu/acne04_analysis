import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class DermnetAcneDataset(Dataset):
    def __init__(self, root_path):
        self.path = root_path
        self.samples = []
        for class_dir in sorted(os.listdir(root_path)):
            class_path = os.path.join(root_path, class_dir)
            if not os.path.isdir(class_path):
                continue
            label = 1 if "acne" in class_dir.lower() else 0
            for fname in sorted(os.listdir(class_path)):
                self.samples.append((os.path.join(class_path, fname), label))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label
