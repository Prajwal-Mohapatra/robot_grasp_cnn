import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
from dataset_loader import CornellGraspDataset
from model import GraspCNN
import os

def draw_grasp_rect(img, grasp, color=(0, 255, 0), label='Prediction'):
    """
    Draw grasp rectangle on image.
    grasp: [x_center, y_center, angle_radians, height, width]
    """
    x, y, angle, h, w = grasp
    angle = float(angle)

    # Compute rectangle corners
    dx = (w / 2) * np.cos(angle)
    dy = (w / 2) * np.sin(angle)
    hx = (h / 2) * np.sin(angle)
    hy = (h / 2) * np.cos(angle)

    # 4 corners
    p1 = (int(x - dx - hx), int(y - dy + hy))
    p2 = (int(x + dx - hx), int(y + dy + hy))
    p3 = (int(x + dx + hx), int(y + dy - hy))
    p4 = (int(x - dx + hx), int(y - dy - hy))

    corners = [p1, p2, p3, p4]
    for i in range(4):
        cv2.line(img, corners[i], corners[(i+1)%4], color, 2)

    cv2.putText(img, label, (int(x)-40, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def visualize_grasp_prediction(index=0, model_path="checkpoints/best_model.pth"):
    """
    Visualize the predicted grasp vs ground truth.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset and model
    dataset = CornellGraspDataset('./data/cornell')
    rgbd, label = dataset[index]
    image = rgbd[:3].permute(1, 2, 0).numpy() * 255.0  # RGB only for display
    image = image.astype(np.uint8).copy()

    model = GraspCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        rgbd = rgbd.unsqueeze(0).to(device)
        pred = model(rgbd).cpu().numpy().squeeze()

    # Draw prediction (green) and ground truth (red)
    draw_grasp_rect(image, pred, color=(0, 255, 0), label='Prediction')
    draw_grasp_rect(image, label.numpy(), color=(255, 0, 0), label='Ground Truth')

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Predicted vs Ground Truth Grasp")
    plt.show()

if __name__ == "__main__":
    visualize_grasp_prediction(index=0)
