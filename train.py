# ===================== train.py =====================
import torch
from torch.utils.data import DataLoader
from dataset_loader import CornellGraspDataset
from model import GraspCNN
import torch.optim as optim
import torch.nn as nn
import os

# custom_collate
def custom_collate(batch):
    return {
        'rgb': torch.stack([item['rgb'] for item in batch]),
        'depth': torch.stack([item['depth'] for item in batch]),
        'grasp': [item['grasp'] for item in batch]  # list of [N_i, 4, 2]
    }

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset & Dataloader
train_dataset = CornellGraspDataset(root='./data/cornell-grasp')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)

# Model, Loss, Optimizer
model = GraspCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)
        grasps = batch['grasp']  # Still on CPU, list of Tensors

        pred = model(rgb, depth)  # shape depends on your model

        # ⚠️ Replace this with actual target processing
        target = torch.zeros_like(pred)  # Dummy placeholder

        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# Save model
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/grasp_cnn.pth")

