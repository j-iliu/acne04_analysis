
import os
import random
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch.distributed as dist


def sample_style_images(dermnet_root, n_total=20, n_acne=2):
    random.seed(42)
    acne_paths = []
    other_paths = []
    
    for class_dir in os.listdir(dermnet_root):
        class_path = os.path.join(dermnet_root, class_dir)
        if not os.path.isdir(class_path):
            continue
        files = [os.path.join(class_path, f) for f in os.listdir(class_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if 'acne' in class_dir.lower():
            acne_paths.extend(files)
        else:
            other_paths.extend(files)
    
    sampled_acne  = random.sample(acne_paths,  min(n_acne, len(acne_paths)))
    sampled_other = random.sample(other_paths, min(n_total - n_acne, len(other_paths)))
    sampled = sampled_acne + sampled_other
    labels  = [True] * len(sampled_acne) + [False] * len(sampled_other)
    
    combined = list(zip(sampled, labels))
    random.shuffle(combined)
    sampled, labels = zip(*combined)
    
    print(f"Sampled {len(sampled_acne)} acne + {len(sampled_other)} other = {len(sampled)} style images")
    return list(sampled), list(labels)


def visualize_style_images(style_paths, style_labels, save_path="style_images.png"):
    n = len(style_paths)
    fig, axes = plt.subplots(n, 1, figsize=(4, 4 * n))
    if n == 1:
        axes = [axes]
    
    for ax, path, is_acne in zip(axes, style_paths, style_labels):
        img = Image.open(path).convert('RGB')
        ax.imshow(img)
        class_name = os.path.basename(os.path.dirname(path))
        border_color = '#ff4444' if is_acne else '#4444ff'
        border_label = 'ACNE' if is_acne else 'normal'
        
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(6)
        
        ax.set_title(f"[{border_label}] {class_name}", 
                    fontsize=10, 
                    color=border_color,
                    fontweight='bold' if is_acne else 'normal')
        ax.set_xticks([])
        ax.set_yticks([])
    
    acne_patch  = mpatches.Patch(color='#ff4444', label='Acne')
    other_patch = mpatches.Patch(color='#4444ff', label='Other')
    fig.legend(handles=[acne_patch, other_patch], loc='upper right', fontsize=10)
    
    plt.suptitle(f"Style Reference Images (n={n})", fontsize=14, fontweight='bold', y=1.001)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved style grid to {save_path}")


def compute_reference_cdf(style_paths):
    all_pixels = [[], [], []]
    
    for path in style_paths:
        img = np.array(Image.open(path).convert('RGB'))
        for c in range(3):
            all_pixels[c].extend(img[:, :, c].flatten().tolist())
    
    cdfs = []
    for c in range(3):
        hist, _ = np.histogram(all_pixels[c], bins=256, range=(0, 256))
        cdf = hist.cumsum().astype(np.float32)
        cdf /= cdf[-1]
        cdfs.append(cdf)
    
    return cdfs


def match_histogram(img_tensor, ref_cdfs):
    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    result = np.zeros_like(img_np)
    
    for c in range(3):
        channel = img_np[:, :, c]
        src_hist, _ = np.histogram(channel.flatten(), bins=256, range=(0, 256))
        src_cdf = src_hist.cumsum().astype(np.float32)
        src_cdf /= src_cdf[-1]
        
        ref_cdf = ref_cdfs[c]
        lut = np.zeros(256, dtype=np.uint8)
        ref_idx = 0
        for src_val in range(256):
            while ref_idx < 255 and ref_cdf[ref_idx] < src_cdf[src_val]:
                ref_idx += 1
            lut[src_val] = ref_idx
        
        result[:, :, c] = lut[channel]
    
    matched = torch.from_numpy(result).permute(2, 0, 1).float() / 255.0
    return matched.to(img_tensor.device)


class HistogramMatcher:
    def __init__(self, dermnet_root=None, ref_cdfs=None, n_total=20, n_acne=2,
                 device='cuda', visualize=True, save_path="style_images.png"):
        self.device = device
        if ref_cdfs is not None:
            self.ref_cdfs = ref_cdfs  # use precomputed
        else:
            style_paths, style_labels = sample_style_images(dermnet_root, n_total=n_total, n_acne=n_acne)
            self.ref_cdfs = compute_reference_cdf(style_paths)
            if visualize:
                visualize_style_images(style_paths, style_labels, save_path=save_path)
        print(f"Rank {device}: HistogramMatcher ready.")

    def __call__(self, img_tensor):
        if img_tensor.dim() == 3:
            return match_histogram(img_tensor, self.ref_cdfs)
        else:
            return torch.stack([match_histogram(img, self.ref_cdfs) for img in img_tensor])

import pickle

def build_histogram_preprocessor(dermnet_root, base_preprocessor, device, rank,
                                  n_total=20, n_acne=2, visualize=True,
                                  save_path="style_images.png"):
    
    CDF_CACHE = "/kaggle/working/ref_cdfs.pkl"
    if os.path.exists(CDF_CACHE):
        with open(CDF_CACHE, 'rb') as f:
            ref_cdfs = pickle.load(f)
        if rank == 0:
            print(f"Loaded existing CDFs from {CDF_CACHE} — skipping recomputation")
    else:
        if rank == 0:
            style_paths, style_labels = sample_style_images(dermnet_root, n_total=n_total, n_acne=n_acne)
            ref_cdfs = compute_reference_cdf(style_paths)
            with open(CDF_CACHE, 'wb') as f:
                pickle.dump(ref_cdfs, f)
            if visualize:
                visualize_style_images(style_paths, style_labels, save_path=save_path)
            print(f"CDFs computed and saved to {CDF_CACHE}")
        if dist.is_initialized():
            dist.barrier()
        with open(CDF_CACHE, 'rb') as f:
            ref_cdfs = pickle.load(f)

    matcher = HistogramMatcher(ref_cdfs=ref_cdfs, device=device)

    def preprocessor(img_tensor):
        matched = matcher(img_tensor.to(device))
        return base_preprocessor(matched)

    return preprocessor
