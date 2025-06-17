# ===================== visualize.py =====================
import matplotlib.pyplot as plt
import numpy as np
import torch

def show_rgb_depth_grasps(rgb, depth, grasp=None, pred_grasp=None):
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.permute(1, 2, 0).cpu().numpy()
    rgb = np.clip(rgb, 0, 1)  # Ensure values are within [0,1]

    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze(0).cpu().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(rgb)
    axs[0].set_title("RGB")
    axs[1].imshow(depth, cmap='gray')
    axs[1].set_title("Depth")

    H, W = rgb.shape[:2]  # Image dimensions

    if grasp is not None:
        for g in grasp:
            g_np = g.cpu().numpy() if isinstance(g, torch.Tensor) else g
            g_np = np.vstack([g_np, g_np[0]])  # Close the grasp rectangle
            axs[0].plot(g_np[:, 0], g_np[:, 1], 'g-')

    if pred_grasp is not None:
        pred_np = pred_grasp.detach().cpu().numpy()
        pred_np = np.clip(pred_np, 0, [W, H])  # Prevent out-of-bounds
        pred_np = np.vstack([pred_np, pred_np[0]])
        axs[0].plot(pred_np[:, 0], pred_np[:, 1], 'r--')

    axs[0].set_xlim(0, W)
    axs[0].set_ylim(H, 0)  # Flip Y-axis to match image coordinates
    axs[1].set_xlim(0, W)
    axs[1].set_ylim(H, 0)

    plt.tight_layout()
    plt.savefig("grasp_output.png")
    plt.show()
