import os
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
import torch
import sys
import torch.distributed as dist
import pickle

def load_vggface2(model_name=None, stage2=False, model_path=None):

    sys.path.insert(0, '/kaggle/working/VGGFace2-pytorch')
    from models.resnet import resnet50
    model=resnet50(num_classes=8631)
    
    path = os.environ.get('VGGFACE_WEIGHT_PATH')
    if not os.path.exists(path):
        print(f"Please set environment variable: export VGGFACE_WEIGHT_PATH='/path/to/weights.pkl'")
        print(f"download weights at https://drive.google.com/file/d/1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU/view")
    with open(path, 'rb') as f:
        weights_np = pickle.load(f,encoding='latin1')
    
    weights_torch = {}
    for key, value in weights_np.items():
        weights_torch[key] = torch.from_numpy(value)
    
    model.load_state_dict(weights_torch)
    
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False
    
    model.fc = nn.Linear(2048, 2)
    model = model.cuda()

    if stage2:
        model = attach_stage2_head(model, model_name, model_path=model_path)
    elif model_path is not None:
        state = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state, strict=True)
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Loaded stage-1 weights from {model_path}")
            
    def vggface2_preprocessing(img_tensor):
        img_255 = img_tensor * 255.0
        img_bgr = img_255.flip(-3)
        mean_bgr = torch.tensor([91.4953, 103.8827, 131.0912]).view(3, 1, 1).to(img_tensor.device)
        return img_bgr - mean_bgr
    
    return model, vggface2_preprocessing
    
def load_resnet50(model_name=None, stage2=False, model_path=None):
    
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False
    
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    model = model.cuda()

    if stage2:
        model = attach_stage2_head(model, model_name, model_path=model_path)
    elif model_path is not None:
        state = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state, strict=True)
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Loaded stage-1 weights from {model_path}")
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    
    return model, normalize

def attach_stage2_head(model, model_name, model_path=None):

    path = model_path if model_path is not None else f"/kaggle/working/{model_name}_stage_1.pt"
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Need checkpoint at '{path}'")

    state = torch.load(path, map_location='cpu')

    is_stage2_checkpoint = 'fc.0.weight' in state
    
    if is_stage2_checkpoint:
        original_in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(original_in_features, 2),
            nn.Linear(2, 2)
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"loading stage 2 model")
            
    else:
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"loading stage 1 model")


    model.load_state_dict(state)
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f"Loaded weights from {path}")
    if not is_stage2_checkpoint:
        original_fc = model.fc
        model.fc = nn.Sequential(
            original_fc,      
            nn.Linear(2, 2)   
        )
        with torch.no_grad():
            model.fc[1].weight.copy_(torch.eye(2))
            model.fc[1].bias.zero_()
    
    for p in model.parameters():
        p.requires_grad = False
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
    for p in model.fc[1:].parameters():
        p.requires_grad = True

    model = model.cuda()
        
    return model
