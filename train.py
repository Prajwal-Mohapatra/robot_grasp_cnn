import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import GraspCNN
from dataset_loader import CornellGraspDataset
import matplotlib.pyplot as plt
import os

# Config
EPOCHS = 50
BATCH_SIZE = 8
LR = 1e-3
PATIENCE = 5
CHECKPOINT_DIR = "checkpoints"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset loading assumes './data/cornell/' inside repo folder
dataset = CornellGraspDataset('./data/cornell')
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model, loss, optimizer
model = GraspCNN().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_loss = float('inf')
epochs_no_improve = 0

train_losses = []
val_losses = []

def iou_like_accuracy(pred, target, threshold=0.25):
    diff = torch.abs(pred - target)
    passed = (diff[:, 0] < 20) & (diff[:, 1] < 20) & (diff[:, 2] < 0.2)
    return passed.float().mean().item()

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for rgbd, label in train_loader:
        rgbd, label = rgbd.to(device), label.to(device)
        output = model(rgbd)
        loss = loss_fn(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss = 0
    acc_total = 0
    with torch.no_grad():
        for rgbd, label in val_loader:
            rgbd, label = rgbd.to(device), label.to(device)
            output = model(rgbd)
            loss = loss_fn(output, label)
            val_loss += loss.item()
            acc_total += iou_like_accuracy(output, label)

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    avg_acc = acc_total / len(val_loader)

    print(f"Epoch {epoch+1:02d}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={avg_acc:.2f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
        print("✅ New best model saved!")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("🛑 Early stopping triggered.")
            break

torch.save(model.state_dict(), "grasp_model_scratch.pth")
print("🎉 Final model saved as grasp_model_scratch.pth")

plt.figure(figsize=(10,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()
