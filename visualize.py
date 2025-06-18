# ===================== visualize.py =====================
import numpy as np
import cv2
import torch

def draw_grasp(image, grasp, color=(0, 255, 0), thickness=2):
    """
    Draw a 6D grasp (x, y, theta, w, h, q) on an image.
    - image: (H, W, 3) numpy array
    - grasp: tensor or array of shape (6,)
    """
    if isinstance(grasp, torch.Tensor):
        grasp = grasp.detach().cpu().numpy()

    x, y, theta, w, h, q = grasp
    theta_rad = np.deg2rad(theta)

    # Rotation matrix
    R = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad),  np.cos(theta_rad)]
    ])

    # Rectangle corners (centered at origin before rotation)
    dx = w / 2
    dy = h / 2
    corners = np.array([
        [-dx, -dy],
        [-dx,  dy],
        [ dx,  dy],
        [ dx, -dy]
    ])

    # Rotate and shift to center (x, y)
    rotated = (R @ corners.T).T
    rotated[:, 0] += x
    rotated[:, 1] += y
    pts = rotated.astype(np.int32).reshape((-1, 1, 2))

    # Draw rotated rectangle
    img = image.copy()
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

    return img
