import os
import argparse
import contextlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from classification_loaders import load_patches

os.environ.setdefault("ROBOFLOW_API_KEY", "***")

from load_models import load_resnet50, load_vggface2
from classification_loaders import load
from histogram_matching import build_histogram_preprocessor


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--acne_ratio", type=float, default=0.5)
    p.add_argument("--use_histogram", action="store_true")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--cache_batch_size", type=int, default=256)
    return p.parse_args()


def auto_name(args):
    if args.name is not None:
        return args.name
    stem = Path(args.model_path).stem
    for suffix in ("_best_stage_1", "_last_stage_1", "_stage_1"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return f"{stem}_skew{args.acne_ratio}_hist{int(args.use_histogram)}"


def cache_key(args):
    return f"skew{args.acne_ratio}_hist{int(args.use_histogram)}"


def cache_features(model, preprocessor, device, split_name, ds, batch_size, out_path):
    if os.path.exists(out_path):
        print(f"  [{split_name}] loading cached features from {out_path}")
        d = torch.load(out_path)
        return d["z"], d["y"]

    print(f"  [{split_name}] caching features -> {out_path}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, desc=f"caching {split_name}", leave=False):
            imgs = preprocessor(imgs.to(device))
            z = model(imgs)
            feats.append(z.cpu())
            labels.append(lbls.cpu())

    z = torch.cat(feats)
    y = torch.cat(labels)
    torch.save({"z": z, "y": y}, out_path)
    print(f"  [{split_name}] saved {z.shape[0]} features (acne_ratio={y.float().mean():.3f})")
    return z, y


def train_fc1(train_z, train_y, val_z, val_y, args, device):
    fc1 = nn.Linear(2, 2).to(device)
    with torch.no_grad():
        fc1.weight.copy_(torch.eye(2))
        fc1.bias.zero_()

    train_z, train_y = train_z.to(device), train_y.to(device)
    val_z, val_y = val_z.to(device), val_y.to(device)

    opt = torch.optim.AdamW(fc1.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = {k: v.clone() for k, v in fc1.state_dict().items()}
    best_epoch = 0

    for epoch in range(args.epochs):
        fc1.train()
        perm = torch.randperm(len(train_z), device=device)
        tloss_sum, tn = 0.0, 0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i : i + args.batch_size]
            out = fc1(train_z[idx])
            loss = crit(out, train_y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tloss_sum += loss.item() * idx.numel()
            tn += idx.numel()
        train_loss = tloss_sum / tn

        fc1.eval()
        with torch.no_grad():
            vout = fc1(val_z)
            val_loss = crit(vout, val_y).item()
            val_acc = (vout.argmax(1) == val_y).float().mean().item()

        flag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in fc1.state_dict().items()}
            best_epoch = epoch
            flag = " *"
        print(f"epoch {epoch:2d}: train={train_loss:.4f} val={val_loss:.4f} val_acc={val_acc:.4f}{flag}")

        if epoch - best_epoch >= args.patience:
            print(f"early stop at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break

    print(f"\nbest val_loss={best_val_loss:.4f} at epoch {best_epoch}")
    return best_state


def main():
    args = parse()
    name = auto_name(args)
    ckey = cache_key(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"=== train_stage2_cached ===")
    print(f"name:         {name}")
    print(f"acne_ratio:   {args.acne_ratio}")
    print(f"use_hist:     {args.use_histogram}")
    print(f"lr/epochs/bs: {args.lr} / {args.epochs} / {args.batch_size}")

    load(version=2)

    use_vggface2 = "vggface2" in args.model_path
    if use_vggface2:
        model, base_preprocessor = load_vggface2(stage2=False)
    else:
        model, base_preprocessor = load_resnet50(stage2=False)

    sd = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    print(f"loaded stage-1 from {args.model_path}")

    if args.use_histogram:
        preprocessor = build_histogram_preprocessor(
            dermnet_root="/kaggle/input/datasets/shubhamgoel27/dermnet/train",
            base_preprocessor=base_preprocessor,
            device=device,
            rank=0,
            visualize=False,
        )
    else:
        preprocessor = base_preprocessor

    _, _, test_soft_ds = load_patches(stage2=False, acne_ratio=args.acne_ratio, jitter_on=False)
    loader = DataLoader(test_soft_ds, batch_size=256, shuffle=False, num_workers=2)
    crit = nn.KLDivLoss(reduction="batchmean")
    
    total, n = 0.0, 0
    with torch.no_grad():
        for imgs, soft_lbls in loader:
            imgs = preprocessor(imgs.to(device))
            log_probs = F.log_softmax(model(imgs), dim=1)
            total += crit(log_probs, soft_lbls.to(device)).item() * imgs.size(0)
            n += imgs.size(0)
    print(f"stage-1 test KLDiv: {total / n:.4f}")

    train_ds, val_ds, test_ds = load_patches(stage2=True, acne_ratio=args.acne_ratio, jitter_on=False)

    cache_dir = "/kaggle/working"
    splits = {"train": train_ds, "val": val_ds, "test": test_ds}

    cached = {}
    for split, ds in splits.items():
        out_path = f"{cache_dir}/cache_{ckey}_{split}.pt"
        cached[split] = cache_features(
            model, preprocessor, device, split, ds, args.cache_batch_size, out_path,
        )

    train_z, train_y = cached["train"]
    val_z, val_y = cached["val"]
    test_z, test_y = cached["test"]

    print(f"\ntrain: n={len(train_z)} acne_frac={train_y.float().mean():.3f}")
    print(f"val:   n={len(val_z)} acne_frac={val_y.float().mean():.3f}")
    print(f"test:  n={len(test_z)} acne_frac={test_y.float().mean():.3f}")

    print(f"\ntraining fc.1")
    best_fc1_state = train_fc1(train_z, train_y, val_z, val_y, args, device)

    fc1 = nn.Linear(2, 2).to(device)
    fc1.load_state_dict(best_fc1_state)
    fc1.eval()
    with torch.no_grad():
        out = fc1(test_z.to(device))
        preds = out.argmax(1)
        acc = (preds == test_y.to(device)).float().mean().item()

    full_state = {}
    for k, v in sd.items():
        if k == "fc.weight":
            full_state["fc.0.weight"] = v
        elif k == "fc.bias":
            full_state["fc.0.bias"] = v
        else:
            full_state[k] = v
    full_state["fc.1.weight"] = best_fc1_state["weight"].cpu()
    full_state["fc.1.bias"] = best_fc1_state["bias"].cpu()

    out_path = f"{cache_dir}/{name}_stage_2.pt"
    torch.save(full_state, out_path)
    print(f"\nsaved stage-2 checkpoint to {out_path}")
    print(f"evaluate with: !python evaluate_acne_model.py --model_path='{out_path}'")
          f"{',--use_histogram' if args.use_histogram else ''})")


if __name__ == "__main__":
    main()
