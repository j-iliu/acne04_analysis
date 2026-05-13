import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Robust hook registration
        self.target_layer.register_forward_hook(self.save_act)
        self.target_layer.register_full_backward_hook(self.save_grad)

    def save_act(self, module, input, output):
        self.activations = output.detach()

    def save_grad(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx=None):
        # Force gradient flow through frozen layers
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass on the specific class
        output[0, class_idx].backward()
        
        if self.gradients is None or self.activations is None:
            return None, class_idx, F.softmax(output, dim=1)[0].detach().cpu().numpy()

        # Compute Grad-CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        
        # Scale to input size
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
        cam = cam.cpu().squeeze().numpy()
        
        # Robust normalization to prevent "blank" plots
        denom = cam.max() - cam.min()
        cam = (cam - cam.min()) / (denom + 1e-8) if denom > 0 else np.zeros_like(cam)
        
        return cam, class_idx, F.softmax(output, dim=1)[0].detach().cpu().numpy()

def visualize_gradcam_batch(gcam, preprocessor, dataloader, device, n=4, class_names=None):
    if class_names is None: class_names = ['Normal', 'Acne']
    
    # Reduced size for better Kaggle rendering
    fig, axes = plt.subplots(n, 2, figsize=(6, 2.0 * n))
    if n == 1: axes = np.expand_dims(axes, axis=0)

    count = 0
    for imgs, labels in dataloader:
        if count >= n: break
        
        img = imgs[0]
        true_idx = labels[0].item()
        
        # Prepare input
        inp = preprocessor(img.unsqueeze(0).to(device))
        cam, pred_idx, probs = gcam.generate(inp)
        
        # Prepare display image
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        
        if cam is not None:
            heatmap = cm.jet(cam)[..., :3]
            result = heatmap * 0.3 + img_np * 0.7
        else:
            result = img_np # Fallback if CAM fails

        # Plotting with label validation color-coding
        title_color = 'green' if true_idx == pred_idx else 'red'
        
        axes[count, 0].imshow(img_np)
        axes[count, 0].set_title(f"True: {class_names[true_idx]}", fontsize=8, color=title_color)
        axes[count, 0].axis('off')
        
        axes[count, 1].imshow(result)
        axes[count, 1].set_title(f"Pred: {class_names[pred_idx]} ({probs[pred_idx]:.2f})", fontsize=8)
        axes[count, 1].axis('off')
        
        count += 1
    
    plt.tight_layout()
    plt.show()
