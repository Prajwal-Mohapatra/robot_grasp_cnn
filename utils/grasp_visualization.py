import cv2
import numpy as np

def draw_grasp_rect(img, grasp, color=(0, 255, 0), label=None, thickness=2):
    """
    Draw a grasp rectangle on the image.

    Parameters:
    - img: RGB image as a NumPy array (HxWx3)
    - grasp: list or array of 5 values [x_center, y_center, angle, height, width]
    - color: BGR color tuple (default green)
    - label: optional text label to draw
    - thickness: line thickness (default 2)
    """
    x, y, angle, h, w = grasp
    x, y, h, w = float(x), float(y), float(h), float(w)
    angle = float(angle)

    # Compute half-dimensions
    dx = (w / 2) * np.cos(angle)
    dy = (w / 2) * np.sin(angle)
    hx = (h / 2) * np.sin(angle)
    hy = (h / 2) * np.cos(angle)

    # 4 corner points of the grasp rectangle
    p1 = (int(x - dx - hx), int(y - dy + hy))
    p2 = (int(x + dx - hx), int(y + dy + hy))
    p3 = (int(x + dx + hx), int(y + dy - hy))
    p4 = (int(x - dx + hx), int(y - dy - hy))

    corners = [p1, p2, p3, p4]

    # Draw rectangle
    for i in range(4):
        cv2.line(img, corners[i], corners[(i + 1) % 4], color, thickness)

    # Optional label
    if label:
        cv2.putText(
            img,
            label,
            (int(x) - 40, int(y) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA
        )

    return img
