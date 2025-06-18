# ===================== train.py =====================
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import GraspCNN
from dataset_loader import CornellGraspDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# --- Helper Functions ---

def get_grasp_center(grasp):
    # grasp: (B, 6) -> (x, y)
    return grasp[:, :2]

def get_closest_grasp(pred, gt_grasps):
    """
    For each prediction, find the closest ground truth grasp rectangle
    (based on grasp center coordinates)
    """
    pred_center = get_grasp_center(pred)  # [B, 2]
    closest_gt = []

    for i in range(pred.shape[0]):
        gt = gt_grasps[i]  # [N, 6]
        gt_centers = gt[:, :2]  # [N, 2]
        dists = torch.norm(gt_centers.to(pred.device) - pred_center[i].unsqueeze(0), dim=1)
        closest = gt[torch.argmin(dists)]
        closest_gt.append(closest)

    return torch.stack(closest_gt)  # [B, 6]

# --- Dataset and Dataloaders ---

train_dataset = CornellGraspDataset(split='train')
val_dataset = CornellGraspDataset(split='val')

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# --- Model, Loss, Optimizer ---

model = GraspCNN(output_dim=6).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# --- Training Loop ---

num_epochs = 10
train_losses, val_losses = [], []

for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        if batch is None:
            continue
        rgb, depth, grasps = batch['rgb'].to(device), batch['depth'].to(device), batch['grasp']

        output = model(rgb, depth)  # [B, 6]
        closest_gt = get_closest_grasp(output, grasps)  # [B, 6]

        loss = criterion(output, closest_gt.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # --- Validation ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            rgb, depth, grasps = batch['rgb'].to(device), batch['depth'].to(device), batch['grasp']

            output = model(rgb, depth)  # [B, 6]
            closest_gt = get_closest_grasp(output, grasps)  # [B, 6]

            loss = criterion(output, closest_gt.to(device))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# --- Plotting Loss ---

plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s', color='red')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
plt.show()

