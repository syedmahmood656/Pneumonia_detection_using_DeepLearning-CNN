import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple
import matplotlib.pyplot as plt


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Works with CXRNet by hooking into the last Conv layer of stage5.

    Usage:
        cam_gen = GradCAM(model, target_layer=model.stage5[-1].block[0])
        cam     = cam_gen.generate(input_tensor, class_idx=12)  # Pneumonia
        overlay = cam_gen.overlay(original_img, cam)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self._activations: torch.Tensor = None
        self._gradients:   torch.Tensor = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """
        Args:
            input_tensor: (1, 1, 224, 224) preprocessed image tensor
            class_idx:    target class index (0–13)

        Returns:
            cam: (224, 224) numpy float32 array in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)        # (1, 14)
        self.model.zero_grad()
        output[0, class_idx].backward()         # Gradient for target class

        # Global average pool gradients over spatial dims → importance weights
        weights = self._gradients.mean(dim=[2, 3], keepdim=True)  # (1, 512, 1, 1)

        # Weighted sum of feature maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, 7, 7)
        cam = F.relu(cam)                        # Only positive contributions
        cam = F.interpolate(cam, size=(224, 224),
                            mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)   # Normalize
        return cam

    def overlay(self, image_np: np.ndarray, cam: np.ndarray,
                alpha: float = 0.45) -> np.ndarray:
        """
        Blend Grad-CAM heatmap over original image.

        Args:
            image_np: (H, W) uint8 grayscale array (original resolution)
            cam:      (224, 224) float in [0, 1]
            alpha:    heatmap transparency (0=invisible, 1=full heatmap)

        Returns:
            blended: (H, W, 3) uint8 BGR image
        """
        H, W = image_np.shape[:2]
        cam_resized = cv2.resize(cam, (W, H))
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        if image_np.ndim == 2:
            orig_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        else:
            orig_bgr = image_np
        return cv2.addWeighted(orig_bgr, 1 - alpha, heatmap, alpha, 0)


def export_gradcam_panel(model, input_tensor, original_np, label_names,
                          output_path: str = 'reports/gradcam_panel.png'):
    """
    Export a 4×4 grid showing Grad-CAM for all 14 classes + original image.
    """
    target_layer = model.stage5[-1].block[0]    # Last Conv2d in stage5
    cam_gen = GradCAM(model, target_layer)

    probs = torch.sigmoid(model(input_tensor)).squeeze().detach().cpu().numpy()
    sorted_classes = np.argsort(probs)[::-1]

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()
    axes[0].imshow(original_np, cmap='gray')
    axes[0].set_title('Original X-Ray', fontsize=9)
    axes[0].axis('off')

    for i, cls_idx in enumerate(sorted_classes[:14]):
        cam = cam_gen.generate(input_tensor.clone(), int(cls_idx))
        overlay = cam_gen.overlay(original_np, cam)
        axes[i + 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[i + 1].set_title(f"{label_names[cls_idx]}\np={probs[cls_idx]:.3f}",
                               fontsize=8)
        axes[i + 1].axis('off')

    plt.suptitle('Grad-CAM — Per-Class Explanations', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Panel saved: {output_path}")