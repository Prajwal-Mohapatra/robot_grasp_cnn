# ===================== visualize.py =====================
from matplotlib import legend
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import torch

def show_rgb_depth_grasps(rgb, depth, grasp=None, pred_grasp=None, save_path=None):
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.permute(1, 2, 0).cpu().numpy()
    rgb = np.clip(rgb, 0, 1)

    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze(0).cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(rgb)
    axs[0].set_title("RGB")
    axs[1].imshow(depth, cmap='gray')
    axs[1].set_title("Depth")

    legend_handles = []

    if grasp is not None:
        for g in grasp:
            g = g.numpy()
            g = np.vstack([g, g[0]])  # Close the grasp rectangle
            axs[0].plot(g[:, 0], g[:, 1], 'b-')
        legend_handles.append(Line2D([0],[0], color = 'blue', lw=1, label='Ground Truth'))

    if pred_grasp is not None:
        pred_grasp = pred_grasp.detach().cpu().numpy()
        pred_grasp = np.vstack([pred_grasp, pred_grasp[0]])
        axs[0].plot(pred_grasp[:, 0], pred_grasp[:, 1], 'y-')
        legend_handles.append(Line2D([0],[0], color = 'yellow', lw=1, label='Prediction'))

    if legend_handles:
      axs[0].legend(handles=legend_handles, loc='lower right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
