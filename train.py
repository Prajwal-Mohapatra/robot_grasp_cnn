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
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

# Custom collate function
def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    return {
        'rgb': torch.stack([item['rgb'] for item in batch]),
        'depth': torch.stack([item['depth'] for item in batch]),
        'pos_grasps': [item['pos_grasps'] for item in batch],
        'neg_grasps': [item['neg_grasps'] for item in batch]
    }

# Function to calculate IoU between two rectangles
def calculate_iou(rect1, rect2):
    """Calculate IoU between two rectangles defined by 4 corner points"""
    try:
        poly1 = Polygon(rect1)
        poly2 = Polygon(rect2)
        
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
            
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        
        if union == 0:
            return 0.0
            
        return intersection / union
    except:
        return 0.0

# Function to get best matching grasp (highest IoU with positive, lowest with negative)
def get_best_grasp_target(pred, pos_grasps, neg_grasps):
    """Find the best positive grasp target for each prediction"""
    batch_targets = []
    batch_ious = []
    
    for i in range(len(pos_grasps)):
        pos = pos_grasps[i]
        neg = neg_grasps[i]
        pred_rect = pred[i].detach().cpu().numpy()
        
        best_iou = 0.0
        best_grasp = None
        
        # Find best positive grasp
        if len(pos) > 0:
            for pos_grasp in pos:
                iou = calculate_iou(pred_rect, pos_grasp.detach().cpu().numpy())
                if iou > best_iou:
                    best_iou = iou
                    best_grasp = pos_grasp
        
        # If no good positive grasp found, use the first positive grasp
        if best_grasp is None and len(pos) > 0:
            best_grasp = pos[0]
            best_iou = calculate_iou(pred_rect, best_grasp.detach().cpu().numpy())
        
        # If still no grasp, create a zero grasp
        if best_grasp is None:
            best_grasp = torch.zeros((4, 2), device=pred.device)
        
        batch_targets.append(best_grasp.to(pred.device))
        batch_ious.append(best_iou)
    
    return torch.stack(batch_targets), torch.tensor(batch_ious)

# Function to calculate grasp success rate
def calculate_grasp_success_rate(pred, pos_grasps, neg_grasps, iou_threshold=0.25):
    """Calculate grasp success rate based on IoU threshold"""
    successes = 0
    total = 0
    
    for i in range(len(pos_grasps)):
        pos = pos_grasps[i]
        pred_rect = pred[i].detach().cpu().numpy()
        
        # Check if prediction overlaps with any positive grasp
        max_iou = 0.0
        if len(pos) > 0:
            for pos_grasp in pos:
                iou = calculate_iou(pred_rect, pos_grasp.detach().cpu().numpy())
                max_iou = max(max_iou, iou)
        
        if max_iou >= iou_threshold:
            successes += 1
        total += 1
    
    return successes / total if total > 0 else 0.0

# Hyperparameters
BATCH_SIZE = 8  # Reduced for stability
EPOCHS = 50
LEARNING_RATE = 0.001
STEP_SIZE = 10
GAMMA = 0.2
VAL_SPLIT = 0.2
IOU_THRESHOLD = 0.25

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
train_success_rates = []
val_success_rates = []
lrs = []

print("Starting training...")

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    running_train_loss = 0.0
    running_train_iou = 0.0
    train_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        if batch is None:
            continue
            
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)
        pos_grasps = batch['pos_grasps']
        neg_grasps = batch['neg_grasps']

        # Forward pass
        pred = model(rgb, depth)
        
        # Get best target grasp for each prediction
        target, ious = get_best_grasp_target(pred, pos_grasps, neg_grasps)
        
        # Calculate loss
        loss = 0.7 * criterion_mse(pred, target) + 0.3 * criterion_smooth(pred, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        running_train_loss += loss.item()
        running_train_iou += ious.mean().item()
        train_batches += 1

    # Calculate average training metrics
    avg_train_loss = running_train_loss / train_batches if train_batches > 0 else 0
    avg_train_iou = running_train_iou / train_batches if train_batches > 0 else 0
    train_losses.append(avg_train_loss)

    # Validation Loop
    model.eval()
    running_val_loss = 0.0
    running_val_iou = 0.0
    val_batches = 0
    val_success_count = 0
    val_total_count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
                
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            pos_grasps = batch['pos_grasps']
            neg_grasps = batch['neg_grasps']

            # Forward pass
            pred = model(rgb, depth)
            
            # Get best target grasp for each prediction
            target, ious = get_best_grasp_target(pred, pos_grasps, neg_grasps)
            
            # Calculate loss
            val_loss = 0.7 * criterion_mse(pred, target) + 0.3 * criterion_smooth(pred, target)
            
            # Calculate success rate
            success_rate = calculate_grasp_success_rate(pred, pos_grasps, neg_grasps, IOU_THRESHOLD)
            
            # Track metrics
            running_val_loss += val_loss.item()
            running_val_iou += ious.mean().item()
            val_success_count += success_rate * len(pos_grasps)
            val_total_count += len(pos_grasps)
            val_batches += 1

    # Calculate average validation metrics
    avg_val_loss = running_val_loss / val_batches if val_batches > 0 else 0
    avg_val_iou = running_val_iou / val_batches if val_batches > 0 else 0
    val_success_rate = val_success_count / val_total_count if val_total_count > 0 else 0
    
    val_losses.append(avg_val_loss)
    val_success_rates.append(val_success_rate)
    lrs.append(optimizer.param_groups[0]['lr'])

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_train_iou:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_val_iou:.4f}")
    print(f"  Val Success Rate: {val_success_rate:.4f}, LR: {lrs[-1]:.6f}")
    print("-" * 50)

    scheduler.step()

# Save model
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/grasp_cnn.pth")
print("✅ Model saved to saved_models/grasp_cnn.pth")

# Model Summary
print("\n" + "="*50)
print("MODEL SUMMARY")
print("="*50)
summary(model, input_size=[(1, 3, 224, 224), (1, 1, 224, 224)])

# Plot Training Results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Loss curves
ax1.plot(range(1, EPOCHS + 1), train_losses, 'b-o', label='Training Loss', markersize=4)
ax1.plot(range(1, EPOCHS + 1), val_losses, 'r-s', label='Validation Loss', markersize=4)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training vs Validation Loss')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Success rate
ax2.plot(range(1, EPOCHS + 1), val_success_rates, 'g-^', label='Validation Success Rate', markersize=4)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Success Rate')
ax2.set_title(f'Validation Success Rate (IoU > {IOU_THRESHOLD})')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Learning rate
ax3.plot(range(1, EPOCHS + 1), lrs, 'purple', marker='o', markersize=4)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Learning Rate')
ax3.set_title('Learning Rate Schedule')
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# Final metrics
final_metrics = [
    f"Final Train Loss: {train_losses[-1]:.4f}",
    f"Final Val Loss: {val_losses[-1]:.4f}",
    f"Final Success Rate: {val_success_rates[-1]:.4f}",
    f"Best Success Rate: {max(val_success_rates):.4f}",
    f"IoU Threshold: {IOU_THRESHOLD}"
]

ax4.text(0.1, 0.9, '\n'.join(final_metrics), transform=ax4.transAxes, fontsize=12,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')
ax4.set_title('Training Summary')

plt.tight_layout()
plt.savefig("training_results.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ Training completed!")
print(f"📊 Final Results:")
print(f"   - Best Success Rate: {max(val_success_rates):.4f}")
print(f"   - Final Success Rate: {val_success_rates[-1]:.4f}")
print(f"   - Final Validation Loss: {val_losses[-1]:.4f}")
