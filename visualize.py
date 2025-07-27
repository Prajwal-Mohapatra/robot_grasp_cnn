#================== visualize.py ====================
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
import math

def rect_from_grasp_param(pred_params, image_size):
    """
    Converts a 6-element prediction vector [x, y, sin(2θ), cos(2θ), w, h] 
    into a 4x2 rectangle of corner points, scaled to the image size.
    """
    if isinstance(pred_params, torch.Tensor):
        pred_params = pred_params.detach().cpu().numpy()
    
    x, y, sin2a, cos2a, w, h = pred_params
    
    # Denormalize using the provided image size
    x, y, w, h = x * image_size, y * image_size, w * image_size, h * image_size
    
    angle = np.arctan2(sin2a, cos2a) / 2.0
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    
    p1 = [x - w/2 * cos_a + h/2 * sin_a, y - w/2 * sin_a - h/2 * cos_a]
    p2 = [x + w/2 * cos_a + h/2 * sin_a, y + w/2 * sin_a - h/2 * cos_a]
    p3 = [x + w/2 * cos_a - h/2 * sin_a, y + w/2 * sin_a + h/2 * cos_a]
    p4 = [x - w/2 * cos_a - h/2 * sin_a, y - w/2 * sin_a + h/2 * cos_a]
    
    return np.array([p1, p2, p3, p4])

def show_rgb_depth_grasps(rgb, depth, gt_grasps=None, pred_params=None, image_size=300, save_path=None):
    """Visualizes RGB, depth, ground truth, and predicted grasps."""
    if isinstance(rgb, torch.Tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        rgb = rgb.cpu() * std + mean
        rgb = rgb.permute(1, 2, 0).numpy()
    rgb = np.clip(rgb, 0, 1)

    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(rgb); axs[0].set_title("RGB with Grasps")
    axs[1].imshow(depth, cmap='gray'); axs[1].set_title("Depth")

    if gt_grasps is not None and len(gt_grasps) > 0:
        for g in gt_grasps:
            g_np = g.cpu().numpy()
            g_closed = np.vstack([g_np, g_np[0]])
            axs[0].plot(g_closed[:, 0], g_closed[:, 1], 'g-', linewidth=2.5)

    if pred_params is not None:
        pred_rect = rect_from_grasp_param(pred_params, image_size)
        rect_closed = np.vstack([pred_rect, pred_rect[0]])
        axs[0].plot(rect_closed[:, 0], rect_closed[:, 1], 'r--', linewidth=2.5)

    legend_handles = [Line2D([0], [0], color='green', lw=2.5, label='Ground Truth'),
                      Line2D([0], [0], color='red', linestyle='--', lw=2.5, label='Prediction')]
    axs[0].legend(handles=legend_handles, loc='upper right')

    for ax in axs: ax.axis('off')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
