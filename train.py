# ===================== train.py =====================
import torch
from torch.utils.data import DataLoader, random_split
from dataset_loader import CornellGraspDataset
from model import GraspCNN
import torch.optim as optim
import torch.nn as nn
import os
import matplotlib.pyplot as plt

# custom_collate
def custom_collate(batch):
    return {
        'rgb': torch.stack([item['rgb'] for item in batch]),
        'depth': torch.stack([item['depth'] for item in batch]),
        'grasp': [item['grasp'] for item in batch]
    }

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001
STEP_SIZE = 5
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
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

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

        target = torch.stack([
            g[0] if g.shape[0] > 0 else torch.zeros((4, 2)) for g in grasps
        ]).to(device)

        loss = criterion(pred, target)
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

            target = torch.stack([
                g[0] if g.shape[0] > 0 else torch.zeros((4, 2)) for g in grasps
            ]).to(device)

            val_loss = criterion(pred, target)
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
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/grasp_cnn.pth")

# Plot Losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), train_losses, 'b-o', label='Training Loss')
plt.plot(range(1, EPOCHS + 1), val_losses, 'r-s', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.grid(True)
plt.legend()

# Annotate LR changes
for i in range(EPOCHS):
    if i == 0 or lrs[i] != lrs[i-1]:
        plt.annotate(f"LR={lrs[i]:.5f}", (i+1, train_losses[i]),
                     textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()
