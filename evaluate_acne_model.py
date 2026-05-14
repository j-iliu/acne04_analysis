import argparse

def evaluate_acne_model(model_path=None, stage2=True, model_name="chud", use_histogram=False):
    from load_models import load_resnet50, load_vggface2
    from classification_loaders import load_patches
    from domain_transfer_dataset import DermnetAcneDataset
    from evaluation import calculate_accuracy
    from classification_dataset import Acne04PatchDataset
    from torch.utils.data import DataLoader, Subset
    from tqdm import tqdm
    import torch

    DERMNET_ROOT = "/kaggle/input/datasets/shubhamgoel27/dermnet"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    use_vggface2 = "vggface2" in (model_path or "")
    if use_vggface2:
        model, preprocessor = load_vggface2(model_name=model_name, stage2=stage2, model_path=model_path)
    else:
        model, preprocessor = load_resnet50(model_name=model_name, stage2=stage2, model_path=model_path)

    model.eval()
    model = model.to(device)

    if use_histogram:
        from histogram_matching import build_histogram_preprocessor
        preprocessor = build_histogram_preprocessor(
            dermnet_root=f"{DERMNET_ROOT}/train",
            base_preprocessor=preprocessor,
            device=device,
            rank=0,
            visualize=False,
        )
        print("Using histogram matching preprocessor")
    else:
        print("Using base preprocessor")

    _, _, test_ds = load_patches(stage2=True, acne_ratio=args.acne_ratio, jitter_on=not use_histogram)

    test_loader = DataLoader(test_ds, batch_size=32, shuffle=True, num_workers=2)

    acne_count   = sum(1 for i in range(len(test_ds)) if test_ds[i][1].item() == 1)
    normal_count = len(test_ds) - acne_count
    print(f"\nACNE04 test: acne={acne_count}, normal={normal_count}, total={len(test_ds)}")

    dermnet_ds  = DermnetAcneDataset(f"{DERMNET_ROOT}/test")
    derm_loader = DataLoader(dermnet_ds, batch_size=32, shuffle=False, num_workers=2)

    derm_acne   = sum(1 for i in range(len(dermnet_ds)) if dermnet_ds[i][1] == 1)
    derm_normal = len(dermnet_ds) - derm_acne
    print(f"DermNet test: acne={derm_acne}, normal={derm_normal}, total={len(dermnet_ds)}")
    print(f"DermNet acne ratio: {derm_acne/len(dermnet_ds):.2%}")

    acne_indices   = [i for i in range(len(dermnet_ds)) if dermnet_ds[i][1] == 1]
    normal_indices = [i for i in range(len(dermnet_ds)) if dermnet_ds[i][1] == 0]
    acne_ds        = Subset(dermnet_ds, acne_indices)
    normal_ds      = Subset(dermnet_ds, normal_indices)
    acne_loader    = DataLoader(acne_ds,   batch_size=32, shuffle=False, num_workers=2)
    normal_loader  = DataLoader(normal_ds, batch_size=32, shuffle=False, num_workers=2)

    print(f"\nAcne-only subset:  {len(acne_ds)} images")
    print(f"Non-acne subset:   {len(normal_ds)} images")

    def get_pred_distribution(loader, model, preprocessor, device):
        pred_acne = pred_normal = 0
        model.eval()
        with torch.no_grad():
            for imgs, labels in tqdm(loader, leave=False):
                imgs  = preprocessor(imgs.to(device))
                preds = model(imgs).argmax(dim=1)
                pred_acne   += (preds == 1).sum().item()
                pred_normal += (preds == 0).sum().item()
        return pred_acne, pred_normal

    print("\nOn DermNet ACNE subset:")
    pa, pn = get_pred_distribution(acne_loader, model, preprocessor, device)
    print(f"  predicted acne={pa}, predicted normal={pn} (should be mostly acne)")

    print("\nOn DermNet NON-ACNE subset:")
    pa, pn = get_pred_distribution(normal_loader, model, preprocessor, device)
    print(f"  predicted acne={pa}, predicted normal={pn} (should be mostly normal)")

    print("\nACNE04 Test Accuracy:")
    calculate_accuracy(model, test_loader, device, preprocessor=preprocessor)

    print("\nDermNet Full Accuracy:")
    calculate_accuracy(model, derm_loader, device, preprocessor=preprocessor)

    print("\nDermNet Acne-only Accuracy:")
    calculate_accuracy(model, acne_loader, device, preprocessor=preprocessor)

    print("\nDermNet Non-acne Accuracy:")
    calculate_accuracy(model, normal_loader, device, preprocessor=preprocessor)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--use_histogram_matching", action='store_true', default=False)
    parser.add_argument("--stage1", action='store_true', default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse() 
    evaluate_acne_model(model_path=args.model_path, stage2=not args.stage1, model_name=args.name, use_histogram=args.use_histogram_matching)
