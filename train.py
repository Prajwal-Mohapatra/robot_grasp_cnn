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
        'grasp': [item['grasp'] for item in batch]  # Keep as list due to variable shape
    }

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001

# Dataset & Dataloader
dataset = CornellGraspDataset(root='./data/cornell-grasp')
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate)

# Model, Loss, Optimizer
model = GraspCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
      rgb = batch['rgb'].to(device)
      depth = batch['depth'].to(device)
      grasps = batch['grasp']  # Leave as list

      for i in range(len(grasps)):
          g = grasps[i].to(device)  # Each g is shape [N_i, 4, 2]
          # use `g` as needed per sample — adapt to your model


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
