# ===================== train.py =====================
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from dataset_loader import CornellGraspDataset
from model import GraspCNN
import torch.optim as optim
from torchinfo import summary
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import math
import numpy as np

# custom_collate
def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    return {
        'rgb': torch.stack([item['rgb'] for item in batch]),
        'depth': torch.stack([item['depth'] for item in batch]),
        'grasp': [item['grasp'] for item in batch]
    }

# Function to get closest ground-truth grasp to the prediction

def rect_from_grasp_param(grasp):
    x, y, theta, w, h, _ = grasp.detach().cpu().numpy()
    x *= 224
    y *= 224
    w *= 224
    h *= 224
    theta *= np.pi
    dx = (w / 2) * math.cos(theta)
    dy = (w / 2) * math.sin(theta)
    hx = (h / 2) * math.sin(theta)
    hy = (h / 2) * -math.cos(theta)
    p1 = [x - dx - hx, y - dy - hy]
    p2 = [x + dx - hx, y + dy - hy]
    p3 = [x + dx + hx, y + dy + hy]
    p4 = [x - dx + hx, y - dy + hy]
    return np.array([p1, p2, p3, p4])

def polygon_from_rect(rect):
    rect_np = rect.detach().cpu().numpy()
    if rect_np.shape == (6,):
        rect_np = rect_from_grasp_param(rect)
    if rect_np.shape != (4, 2):
        raise ValueError(f"Invalid rect shape for polygon: got {rect_np.shape}, expected (4, 2)")
    return Polygon(rect_np)

def compute_iou(poly1, poly2):
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0.0

def compute_angle(rect):
    if rect.shape != (4, 2):
        print(f"[Warning] compute_angle received invalid rect shape: {rect.shape}")
        return torch.tensor(0.0, device=rect.device)
    dx = rect[1] - rect[0]
    return torch.atan2(dx[1], dx[0]) / torch.pi

def get_closest_grasp(pred, grasps):
    """
    Find the closest ground truth grasp to each predicted grasp vector using L2 distance.
    Assumes pred is [B, 6] and grasps is list of list-of-[4, 2] tensors or [6] tensors.
    """
    batch_targets = []

    for i in range(len(grasps)):
        gt_rects = grasps[i]
        if len(gt_rects) == 0:
            batch_targets.append(torch.zeros(6, device=pred.device))
            continue

        # Convert all ground truth grasps to vector format [6]
        gt_vecs = []
        for g in gt_rects:
            if g.shape == (4, 2):  # Convert rectangle to vector
                g = g.to(pred.device)
                center = g.mean(dim=0)
                dx = g[1] - g[0]
                dy = g[2] - g[1]
                width = torch.norm(dx)
                height = torch.norm(dy)
                theta = torch.atan2(dx[1], dx[0]) / torch.pi  # Normalize
                grasp_vec = torch.tensor([
                    center[0] / 224,
                    center[1] / 224,
                    theta,
                    width / 224,
                    height / 224,
                    1.0
                ], device=pred.device)
                gt_vecs.append(grasp_vec)
            elif g.shape == (6,):
                gt_vecs.append(g.to(pred.device))

        if len(gt_vecs) == 0:
            batch_targets.append(torch.zeros(6, device=pred.device))
            continue

        gt_stack = torch.stack(gt_vecs)  # [N, 6]
        pred_vec = pred[i]  # [6]

        dists = torch.norm(gt_stack - pred_vec.unsqueeze(0), dim=1)  # [N]
        closest = gt_stack[dists.argmin()]
        batch_targets.append(closest)

    return torch.stack(batch_targets)  # [B, 6]


# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
STEP_SIZE = 10
GAMMA = 0.5
VAL_SPLIT = 0.2

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset & DataLoader
full_dataset = CornellGraspDataset(root='./data/cornell-grasp')
val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate)

# Model, Loss, Optimizer, Scheduler
model = GraspCNN().to(device)
criterion_mse = nn.MSELoss()
criterion_smooth = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Tracking
train_losses = []
val_losses = []
lrs = []

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    running_train_loss = 0
    for batch in train_loader:
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)
        grasps = batch['grasp']
        pred = model(rgb, depth)
        target = get_closest_grasp(pred, grasps).to(device)
        #loss = 0.7 * criterion_mse(pred, target) + 0.3 * criterion_smooth(pred, target)
        loss = criterion_mse(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()
    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation Loop
    model.eval()
    running_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            grasps = batch['grasp']
            pred = model(rgb, depth)
            target = get_closest_grasp(pred, grasps).to(device)
            #val_loss = 0.7 * criterion_mse(pred, target) + 0.3 * criterion_smooth(pred, target)
            val_loss = criterion_mse(pred, target)
            running_val_loss += val_loss.item()
    avg_val_loss = running_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    lrs.append(optimizer.param_groups[0]['lr'])
    print(f"Epoch {epoch+1}/{EPOCHS}, "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"LR: {lrs[-1]:.6f}")
    scheduler.step()

# Save model
os.makedirs("outputs/saved_models", exist_ok=True)
torch.save(model.state_dict(), "outputs/saved_models/grasp_cnn.pth")

#Model Summary
summary(model)

# Plot Losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), train_losses, 'b-o', label='Training Loss')
plt.plot(range(1, EPOCHS + 1), val_losses, 'r-s', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("outputs/loss_curve.png")
plt.show()
