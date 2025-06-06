# ===================== train.py =====================
import torch
from torch.utils.data import DataLoader
from dataset_loader import CornellGraspDataset
from model import GraspCNN
import torch.optim as optim
import torch.nn as nn
import os

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001

# Dataset & Dataloader
dataset = CornellGraspDataset(root='./data/cornell')
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, Loss, Optimizer
model = GraspCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        rgb = batch['rgb']
        depth = batch['depth']
        target = batch['grasps'][:, 0]  # Use only one grasp per sample

        pred = model(rgb, depth)

        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# Save model
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/grasp_cnn.pth")
