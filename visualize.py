# ===================== visualize.py =====================
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

def show_rgb_depth_grasps(rgb, depth, grasp=None, pred_grasp=None, save_path=None,
                          original_size=(480, 640), resized_size=(224, 224)):
    """
    Visualize RGB, depth, ground truth grasp rectangles, and predicted grasp.

    Args:
        rgb: Tensor [3, H, W] or np array [H, W, 3]
        depth: Tensor [1, H, W] or np array [H, W]
        grasp: List of torch tensors [4, 2] (ground truth rectangles)
        pred_grasp: Tensor [4, 2] (predicted grasp rectangle)
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
        pred_grasp = pred_grasp.detach().cpu().numpy() if isinstance(pred_grasp, torch.Tensor) else pred_grasp
        pred_grasp = scale_coords(pred_grasp)
        pred_grasp = np.vstack([pred_grasp, pred_grasp[0]])  # Close rectangle
        axs[0].plot(pred_grasp[:, 0], pred_grasp[:, 1], 'y-', linewidth=2)
        legend_handles.append(Line2D([0], [0], color='yellow', lw=2, label='Prediction'))

    if legend_handles:
        axs[0].legend(handles=legend_handles, loc='lower right')

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
