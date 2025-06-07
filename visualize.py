# ===================== visualize.py =====================
import matplotlib.pyplot as plt

def show_rgb_depth_grasps(rgb, depth, grasp=None, pred_grasp=None):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(rgb.permute(1, 2, 0))
    axs[0].set_title("RGB")
    axs[1].imshow(depth.squeeze(0), cmap='gray')
    axs[1].set_title("Depth")

    if grasp is not None:
        for g in grasps:
            g = g.numpy()
            axs[0].plot([p[0] for p in g + [g[0]]], [p[1] for p in g + [g[0]]], 'g-')
    if pred_grasp is not None:
        pred_grasp = pred_grasp.detach().numpy()
        axs[0].plot([p[0] for p in pred_grasp + [pred_grasp[0]]], [p[1] for p in pred_grasp + [pred_grasp[0]]], 'r--')

    plt.show()
