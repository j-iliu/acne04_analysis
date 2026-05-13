
import os
import torch
import json
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import math
import random
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import v2 as T
import torchvision.transforms.functional as TF
from roboflow import Roboflow
from PIL import Image
import torch.distributed as dist

os.environ["ROBOFLOW_API_KEY"] = "2tcxpa2qfGVefGRsodVE"

class Acne04PatchDataset(torch.utils.data.Dataset):
    
    def __init__(self, source, 
                is_train=False,
                canvas_size=224,
                num_classes=2,
                center_range=(0.35, 0.65),
                per_tile_rotation_deg=5.0,
                composition_rotation_deg=5.0,
                pre_rotation_pad_factor=1.25,
                post_rotation_pad_factor=1.25,
                flip_p=0.5,
                interpolation=TF.InterpolationMode.BILINEAR,
                mosaic = True,
                stage2 = False,
                patches_cache = None,
                acne_ratio = 0.5,
                jitter_on = True,
                patch_input=None,
                ):
        self.source = source
        self.is_train = is_train
        self.jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
        self.jitter_on=jitter_on
        self.canvas_size = canvas_size
        self.center_range = center_range
        self.per_tile_rotation_deg = per_tile_rotation_deg
        self.composition_rotation_deg = composition_rotation_deg
        self.flip_p = flip_p
        self.interpolation = interpolation
        self.mosaic = mosaic
        self.stage2 = stage2
        self.patches_cache = patches_cache
        self.patch_input = patch_input
        self.pre_rotation_pad_factor = pre_rotation_pad_factor
        self.post_rotation_pad_factor = post_rotation_pad_factor
        self.acne_ratio = acne_ratio
        
        if patches_cache is not None and os.path.exists(patches_cache):
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Loading cached patches from {patches_cache}")
            self.patches = torch.load(patches_cache, weights_only=False)
        elif patch_input is not None and os.path.exists(patch_input):
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Loading from kaggle at {patch_input}")
            self.patches = torch.load(patch_input, weights_only=False)
        else:
            self._create_patches()
            if patches_cache is not None:
                self._save_patches(patches_cache)
        self.shuffle_mosaic()
        self.mosaic = mosaic
        self.stage2 = stage2

    def __len__(self):
        return len(self.base)

    def _save_patches(self, path):
        if not dist.is_initialized() or dist.get_rank() == 0:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            torch.save(self.patches, path)
            print(f"Saved {len(self.patches)} patches to {path}")

    def _create_patches(self):
        
        boxes = []
        labels = []
        leftover = 0 

        img_WH = self.source[0][0].shape[1]
        
        for img_idx, (img, targets) in enumerate(tqdm(self.source, desc="Making patches")):
            
            total_rgb = [0, 0, 0]
            total_pixels = 0
            ws = []
            hs = []
            acne_patches = []
           
            for ann in targets:
                
                x, y, w, h = ann['bbox']
                
                w = int(w * 1.15)
                h = int(h * 1.15)
            
                ws.append(w)
                hs.append(h)
                
                sums = torch.sum(img[:, y:y+h, x:x+w], dim=(1, 2))
                total_rgb += np.array(sums)
        
                total_pixels += w * h
                acne_patches.append([x, y, x + w, y + h])

            amt_guesses=max(800, len(ws) * 100)
                
            avg_w, std_w = np.mean(ws), np.std(ws)
            avg_h, std_h = np.mean(hs), np.std(hs)
            
            # [x1, y1 -> [0, 0 x2 - x1, y2 - x1] + [x1, y1, x1, y1] --> [x1, y1, x2, y2] 
            guesses_xyxy = np.random.randint(0, img_WH, (amt_guesses, 2))
            to_x2y2 = np.hstack([np.zeros((amt_guesses, 2), dtype=int), np.random.normal([avg_w, avg_h], [std_w, std_h], (amt_guesses, 2)).astype(int)])
            guesses_xyxy = np.hstack([guesses_xyxy, guesses_xyxy])    
            guesses_xyxy = guesses_xyxy + to_x2y2
            
            guesses_xyxy[:, 2] = np.minimum(guesses_xyxy[:, 2], img_WH)
            guesses_xyxy[:, 3] = np.minimum(guesses_xyxy[:, 3], img_WH)
            
            #removes where x1 > x2 or y1 > y2
            guesses_xyxy = guesses_xyxy[(guesses_xyxy[:, 2] > guesses_xyxy[:, 0]) & (guesses_xyxy[:, 3] > guesses_xyxy[:, 1])]
           
            avg_tensor = torch.tensor(total_rgb / total_pixels)
            
            img_norm = img / (torch.norm(img, dim=0, keepdim=True) + 1e-6)
            avg_norm = (avg_tensor / (torch.norm(avg_tensor) + 1e-6)).view(3, 1, 1)
            mask = torch.linalg.norm(img_norm - avg_norm, dim=0) > 0.1
        
            sum_area_tbl = mask.to(torch.int32).cumsum(0).cumsum(1)
            sum_area_tbl = np.pad(sum_area_tbl, ((1,0),(1,0)))
        
            x1, y1, x2, y2 = guesses_xyxy.T
            mask_violation_count = (sum_area_tbl[y2, x2] - 
                                    sum_area_tbl[y1, x2] - 
                                    sum_area_tbl[y2, x1] + 
                                    sum_area_tbl[y1, x1])
            
            violated_mask = mask_violation_count > 0
        
            acne_patches = np.array(acne_patches)
            
            if len(acne_patches):
                ix1 = np.maximum(x1[:, None], acne_patches[:, 0])
                iy1 = np.maximum(y1[:, None], acne_patches[:, 1])
                ix2 = np.minimum(x2[:, None], acne_patches[:, 2])
                iy2 = np.minimum(y2[:, None], acne_patches[:, 3])
                inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
                overlap_acne = inter.any(axis=1)
            else:
                overlap_acne = np.zeros(amt_guesses, dtype=bool)
            
            guesses_xyxy = guesses_xyxy[~(overlap_acne | violated_mask)]
            
            #how much normal skin samples to collect -- limited by usable patches
            samples = np.minimum(int(len(acne_patches) * 2)  + leftover, guesses_xyxy.shape[0])
            #how much more normal skin needed
            leftover += int(len(ws) * 15.67) - samples
            idx = random.sample(range(0, guesses_xyxy.shape[0]), samples)
            normal_patches = guesses_xyxy[idx]
            
            img_path = os.path.join(self.source.root, self.source.coco.imgs[self.source.ids[img_idx]]['file_name'])

            for patch in acne_patches:
                x1, y1, x2, y2 = int(patch[0]), int(patch[1]), int(patch[2]), int(patch[3])
                w, h = x2 - x1, y2 - y1

                boxes.append((img_path, x1, y1, x2, y2))
                labels.extend([1])

            for patch in normal_patches:
                x1, y1, x2, y2 = int(patch[0]), int(patch[1]), int(patch[2]), int(patch[3])
                w, h = x2 - x1, y2 - y1
                if w > 0 and h > 0 and max(w/h, h/w) <= 2.5:
                    boxes.append((img_path, x1, y1, x2, y2))
                    labels.extend([0])
                else:
                    leftover += 1
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"short {leftover} normal patches")
        
        self.patches = list(zip(boxes, labels))

            
    def _create_mosaic(self, patches, labels):
        patches = list(patches)
        labels = [int(l) for l in labels]

    
        device = patches[0].device
        dtype = patches[0].dtype
        H = W = self.canvas_size
    
        center_x = int(round(torch.empty(1).uniform_(*self.center_range).item() * W))
        center_y = int(round(torch.empty(1).uniform_(*self.center_range).item() * H))
    
        tile_xyhw = [
            (0,        0,        center_x,     center_y),
            (center_x, 0,        W - center_x, center_y),
            (0,        center_y, center_x,     H - center_y),
            (center_x, center_y, W - center_x, H - center_y),
        ]
    
        patch_means = [p.mean(dim=(1, 2), keepdim=True) for p in patches]
        global_mean = torch.stack([m.flatten() for m in patch_means]).mean(dim=0).tolist()
    
        mosaic = torch.zeros((3, H, W), device=device, dtype=dtype)
        class_map = torch.zeros((H, W), device=device, dtype=torch.int16)
    
        for (tile_x, tile_y, tile_w, tile_h), patch, label, current_patch_mean in zip(
            tile_xyhw, patches, labels, patch_means
        ):
            if tile_h == 0 or tile_w == 0:
                continue
    
            patch_height, patch_width = patch.shape[-2:]
            scale = max(tile_h / patch_height, tile_w / patch_width)
            resized = TF.resize(
                patch,
                [max(1, int(math.ceil(patch_height * scale))),
                 max(1, int(math.ceil(patch_width  * scale)))],
                interpolation=self.interpolation, antialias=True,
            )
            top  = (resized.shape[-2] - tile_h) // 2
            left = (resized.shape[-1] - tile_w) // 2
            tile = resized[:, top:top + tile_h, left:left + tile_w]
    
            if torch.rand(1).item() < self.flip_p:
                tile = torch.flip(tile, dims=[-1])
            if torch.rand(1).item() < self.flip_p:
                tile = torch.flip(tile, dims=[-2])
    
            tile_angle = torch.empty(1).uniform_(
                -self.per_tile_rotation_deg, self.per_tile_rotation_deg
            ).item()
            tile = TF.rotate(
                tile, tile_angle, interpolation=self.interpolation,
                expand=False, fill=current_patch_mean.flatten().tolist(),
            )
            tile_class = TF.rotate(
                torch.full((1, tile_h, tile_w), label, dtype=torch.float32, device=device),
                tile_angle, interpolation=TF.InterpolationMode.NEAREST,
                expand=False, fill=-1.0,
            ).squeeze(0).to(torch.int16)
    
            mosaic[:, tile_y:tile_y + tile_h, tile_x:tile_x + tile_w] = tile
            class_map[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w] = tile_class
    
        if self.composition_rotation_deg > 0:
            to_pad_H = int(math.ceil(H * (self.post_rotation_pad_factor - 1.0) / 2.0))
            to_pad_W = int(math.ceil(W * (self.post_rotation_pad_factor - 1.0) / 2.0))
            padded_H, padded_W = H + 2 * to_pad_H, W + 2 * to_pad_W
    
            tl_mean, tr_mean, bl_mean, br_mean = patch_means
            mosaic_padded = torch.empty((3, padded_H, padded_W), device=device, dtype=dtype)
            mosaic_padded[:, :to_pad_H + center_y, :to_pad_W + center_x] = tl_mean
            mosaic_padded[:, :to_pad_H + center_y,  to_pad_W + center_x:] = tr_mean
            mosaic_padded[:,  to_pad_H + center_y:, :to_pad_W + center_x] = bl_mean
            mosaic_padded[:,  to_pad_H + center_y:,  to_pad_W + center_x:] = br_mean
            mosaic_padded[:, to_pad_H:to_pad_H + H, to_pad_W:to_pad_W + W] = mosaic
    
            class_map_padded = torch.zeros((padded_H, padded_W), device=device, dtype=torch.int16)
            class_map_padded[to_pad_H:to_pad_H + H, to_pad_W:to_pad_W + W] = class_map
    
            global_angle = torch.empty(1).uniform_(
                -self.composition_rotation_deg, self.composition_rotation_deg
            ).item()
            mosaic_padded = TF.rotate(
                mosaic_padded, global_angle, interpolation=self.interpolation,
                expand=False, fill=global_mean,
            )
            class_map_padded = TF.rotate(
                class_map_padded.unsqueeze(0).float(), global_angle,
                interpolation=TF.InterpolationMode.NEAREST, expand=False, fill=-1.0,
            ).squeeze(0).to(torch.int16)
    
            top_crop  = (padded_H - H) // 2
            left_crop = (padded_W - W) // 2
            mosaic    = mosaic_padded[:, top_crop:top_crop + H, left_crop:left_crop + W]
            class_map = class_map_padded[top_crop:top_crop + H, left_crop:left_crop + W]
    
        valid_mask = class_map.flatten() >= 0
        valid_pixels = class_map.flatten()[valid_mask].to(torch.long)
        counts = torch.bincount(valid_pixels, minlength=2).float()
        soft_label = counts / counts.sum()

        if self.jitter_on:
            mosaic = self.jitter(mosaic)
        
        return mosaic, soft_label

    def shuffle_mosaic(self):

        acne = [patch for patch in self.patches if patch[1] == 1]
        normal = [patch for patch in self.patches if patch[1] == 0]
        
        if not self.stage2:    

            min_length = min(len(acne), len(normal))
            min_length -= min_length % 2
            acne = acne[:min_length]
            normal = normal[:min_length]
            combined = acne + normal  
            random.shuffle(combined)
            grouped = [combined[i:i + 4] for i in range(0, len(combined), 4)]
            self.base = grouped
        else:
            acne_ratio = self.acne_ratio
            amt_mixed_mosaics  = len(acne) // 4  # use all acne
            amt_normal_mosaics = int(amt_mixed_mosaics * (1 - acne_ratio) / acne_ratio)
        
            # check we have enough normals
            normals_needed = amt_mixed_mosaics * 1 + amt_normal_mosaics * 4  # 1 per mixed (worst case), 4 per pure
            if normals_needed > len(normal):
                # scale down amt_normal_mosaics to fit
                normals_for_pure = len(normal) - amt_mixed_mosaics * 1
                amt_normal_mosaics = max(0, normals_for_pure // 4)
        
            acne_copy   = list(acne)
            normal_copy = list(normal)
            random.shuffle(acne_copy)
            random.shuffle(normal_copy)
            grouped = []
        
            # mixed mosaics first (reserve normals for them)
            normals_reserved = normal_copy[:amt_mixed_mosaics]   # 1 per mixed worst case
            normals_for_pure = normal_copy[amt_mixed_mosaics:]
            normal_copy = normals_reserved
        
            for _ in range(amt_mixed_mosaics):
                n_acne = random.choice([3, 4])
                n_norm = 4 - n_acne
                if len(acne_copy) < n_acne:
                    break
                mosaic = [acne_copy.pop() for _ in range(n_acne)]
                for _ in range(n_norm):
                    if normal_copy:
                        mosaic.append(normal_copy.pop())
                    elif acne_copy:
                        mosaic.append(acne_copy.pop())
                if len(mosaic) == 4:
                    random.shuffle(mosaic)
                    grouped.append(mosaic)
        
            for i in range(amt_normal_mosaics):
                mosaic = normals_for_pure[i*4:(i+1)*4]
                if len(mosaic) == 4:
                    grouped.append(mosaic)
        
            actual_acne   = sum(1 for g in grouped if any(item[1]==1 for item in g))
            actual_normal = sum(1 for g in grouped if all(item[1]==0 for item in g))
        
            random.shuffle(grouped)
            self.base = grouped
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Target ratio: {acne_ratio:.0%} | Actual: {actual_acne/(actual_acne+actual_normal):.0%} | Acne mosaics: {actual_acne} | Normal mosaics: {actual_normal}")
        
    def __getitem__(self, idx):
        patches_batch = self.base[idx]
        boxes_list, labels_list = zip(*patches_batch)
        patches_list = tuple(self._load_patch(b) for b in boxes_list)

        if self.mosaic:
            mosaic_img, soft_label = self._create_mosaic(patches_list, labels_list)
            if self.stage2:
                has_acne = any(label == 1 for label in labels_list)
                # hard_label = torch.zeros(2)
                # hard_label[1 if has_acne else 0] = 1.0
                hard_label = torch.tensor(1 if has_acne else 0, dtype=torch.long)
                return mosaic_img, hard_label
            return mosaic_img, soft_label
        else:
            to_pick = np.random.randint(0,4)
            single_patch = patches_list[to_pick]
            single_label = labels_list[to_pick]
            single_patch = TF.resize(single_patch, [224, 224], 
                             interpolation=self.interpolation, antialias=True)

            if self.stage2:
                return single_patch, torch.tensor(single_label, dtype=torch.long)
            soft_label = torch.zeros(2)
            soft_label[single_label] = 1.0
            return single_patch, soft_label
            
    def _load_patch(self, box_meta):
        img_path, x1, y1, x2, y2 = box_meta
        img = Image.open(img_path).convert('RGB')
        patch = img.crop((x1, y1, x2, y2))
        return TF.to_tensor(patch)
