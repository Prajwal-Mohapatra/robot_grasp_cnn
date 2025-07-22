import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
import math

def rect_from_grasp_param(grasp):
    """Converts (6,) grasp param [x, y, θ, w, h, q] to 4 corner points."""
    if isinstance(grasp, torch.Tensor):
        grasp = grasp.detach().cpu().numpy()
    x, y, theta, w, h, _ = grasp

    # Denormalize assuming image size 224x224
    x *= 224
    y *= 224
    w *= 224
    h *= 224
    theta *= np.pi  # Convert back from [-1, 1] to radians

    dx = (w / 2) * np.cos(theta)
    dy = (w / 2) * np.sin(theta)
    hx = (h / 2) * np.sin(theta)
    hy = (h / 2) * -np.cos(theta)

    p1 = [x - dx - hx, y - dy - hy]
    p2 = [x + dx - hx, y + dy - hy]
    p3 = [x + dx + hx, y + dy + hy]
    p4 = [x - dx + hx, y - dy + hy]

    return np.array([p1, p2, p3, p4])

def show_rgb_depth_grasps(rgb, depth, grasp=None, pred_grasp=None, save_path=None,
                          original_size=(480, 640), resized_size=(224, 224)):
    """
    Visualize RGB, depth, ground truth grasp rectangles, and predicted grasp.

    Args:
        rgb: Tensor [3, H, W] or np array [H, W, 3]
        depth: Tensor [1, H, W] or np array [H, W]
        grasp: List of torch tensors [4, 2] (ground truth rectangles)
        pred_grasp: Tensor [6] (predicted grasp vector)
        save_path: If set, saves the figure to the path
    """

    # Convert tensors to numpy
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.permute(1, 2, 0).cpu().numpy()
    rgb = np.clip(rgb, 0, 1)

    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()

    # Compute scaling factors
    h_ratio = resized_size[0] / original_size[0]
    w_ratio = resized_size[1] / original_size[1]

    def scale_coords(coords):
        return coords * np.array([w_ratio, h_ratio])

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(rgb)
    axs[0].set_title("RGB")
    axs[1].imshow(depth, cmap='gray')
    axs[1].set_title("Depth")

    legend_handles = []

    # Plot ground truth grasp rectangles
    if grasp is not None:
        for g in grasp:
            g = g.cpu().numpy() if isinstance(g, torch.Tensor) else g
            g = scale_coords(g)
            g = np.vstack([g, g[0]])  # Close rectangle
            axs[0].plot(g[:, 0], g[:, 1], 'b-', linewidth=2)
        legend_handles.append(Line2D([0], [0], color='blue', lw=2, label='Ground Truth'))

    # Plot predicted grasp
    if pred_grasp is not None:
        rect = rect_from_grasp_param(pred_grasp)  # Convert [6] to [4,2]
        rect = scale_coords(rect)
        rect = np.vstack([rect, rect[0]])  # Close rectangle
        axs[0].plot(rect[:, 0], rect[:, 1], 'y-', linewidth=2)
        legend_handles.append(Line2D([0], [0], color='yellow', lw=2, label='Prediction'))

    if legend_handles:
        axs[0].legend(handles=legend_handles, loc='lower right')

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
