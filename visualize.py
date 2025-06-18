# ===================== visualize.py =====================
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

    if grasp is not None:
        for g in grasp:
            g = g.numpy()
            g = np.vstack([g, g[0]])  # Close the grasp rectangle
            axs[0].plot(g[:, 0], g[:, 1], 'g-')

    if pred_grasp is not None:
        pred_grasp = pred_grasp.detach().cpu().numpy()
        pred_grasp = np.vstack([pred_grasp, pred_grasp[0]])
        axs[0].plot(pred_grasp[:, 0], pred_grasp[:, 1], 'r--')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
