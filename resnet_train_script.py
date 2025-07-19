# ===================== train.py =====================
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
#from dataset_loader import CornellGraspDataset
from loader import CornellGraspDataset
from model import create_grasp_model, grasp_params_to_rectangle, rectangle_to_grasp_params
import torch.optim as optim
from torchinfo import summary
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
import math

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

# Function to convert rectangle coordinates to grasp parameters
def convert_rectangles_to_params(rectangles):
    """Convert rectangle coordinates to [x, y, θ, width] format"""
    batch_params = []
    
    for rect in rectangles:
        if len(rect) == 0:
            # Default grasp if no rectangles
            batch_params.append([112.0, 112.0, 0.0, 50.0])  # center of 224x224 image
            continue
            
        # Take first rectangle if multiple exist
        if len(rect) > 0:
            corners = rect[0] if isinstance(rect, list) else rect
            corners = corners.detach().cpu().numpy() if torch.is_tensor(corners) else corners
            
            # Calculate center
            center_x = np.mean(corners[:, 0])
            center_y = np.mean(corners[:, 1])
            
            # Calculate width (distance between adjacent corners)
            width = np.linalg.norm(corners[1] - corners[0])
            
            # Calculate angle
            edge_vector = corners[1] - corners[0]
            theta = np.arctan2(edge_vector[1], edge_vector[0])
            
            # Ensure theta is in [-π/2, π/2]
            if theta > math.pi / 2:
                theta -= math.pi
            elif theta < -math.pi / 2:
                theta += math.pi
            
            batch_params.append([center_x, center_y, theta, width])
        else:
            batch_params.append([112.0, 112.0, 0.0, 50.0])
    
    return torch.tensor(batch_params, dtype=torch.float32)

# Function to get best matching grasp target
def get_best_grasp_target(pred_params, pos_grasps, neg_grasps):
    """Find the best positive grasp target for each prediction"""
    batch_targets = []
    batch_ious = []
    
    # Convert predictions to rectangles for IoU calculation
    pred_rectangles = grasp_params_to_rectangle(pred_params)
    
    for i in range(len(pos_grasps)):
        pos = pos_grasps[i]
        pred_rect = pred_rectangles[i].detach().cpu().numpy()
        
        best_iou = 0.0
        best_params = None
        
        # Find best positive grasp
        if len(pos) > 0:
            for pos_grasp in pos:
                pos_rect = pos_grasp.detach().cpu().numpy()
                iou = calculate_iou(pred_rect, pos_rect)
                if iou > best_iou:
                    best_iou = iou
                    best_params = rectangle_to_grasp_params(pos_grasp.unsqueeze(0))[0]
        
        # If no good positive grasp found, convert the first positive grasp
        if best_params is None and len(pos) > 0:
            best_params = rectangle_to_grasp_params(pos[0].unsqueeze(0))[0]
            best_iou = calculate_iou(pred_rect, pos[0].detach().cpu().numpy())
        
        # If still no grasp, use prediction as target (will result in zero loss)
        if best_params is None:
            best_params = pred_params[i].detach()
        
        batch_targets.append(best_params.to(pred_params.device))
        batch_ious.append(best_iou)
    
    return torch.stack(batch_targets), torch.tensor(batch_ious)

# Function to calculate grasp success rate
def calculate_grasp_success_rate(pred_params, pos_grasps, neg_grasps, iou_threshold=0.25):
    """Calculate grasp success rate based on IoU threshold"""
    successes = 0
    total = 0
    
    # Convert predictions to rectangles
    pred_rectangles = grasp_params_to_rectangle(pred_params)
    
    for i in range(len(pos_grasps)):
        pos = pos_grasps[i]
        pred_rect = pred_rectangles[i].detach().cpu().numpy()
        
        # Check if prediction overlaps with any positive grasp
        max_iou = 0.0
        if len(pos) > 0:
            for pos_grasp in pos:
                pos_rect = pos_grasp.detach().cpu().numpy()
                iou = calculate_iou(pred_rect, pos_rect)
                max_iou = max(max_iou, iou)
        
        if max_iou >= iou_threshold:
            successes += 1
        total += 1
    
    return successes / total if total > 0 else 0.0

# Custom loss function for grasp parameters
class GraspLoss(nn.Module):
    def __init__(self, pos_weight=2.0, angle_weight=0.5):
        super(GraspLoss, self).__init__()
        self.pos_weight = pos_weight
        self.angle_weight = angle_weight
        self.mse = nn.MSELoss()
        self.smooth_l1 = nn.SmoothL1Loss()
    
    def forward(self, pred, target):
        # pred, target: [batch_size, 4] -> [x, y, θ, width]
        
        # Position loss (x, y)
        pos_loss = self.mse(pred[:, :2], target[:, :2])
        
        # Angle loss (θ) - handle circular nature
        angle_diff = pred[:, 2] - target[:, 2]
        # Normalize angle difference to [-π, π]
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        angle_loss = self.mse(angle_diff, torch.zeros_like(angle_diff))
        
        # Width loss
        width_loss = self.smooth_l1(pred[:, 3], target[:, 3])
        
        # Combine losses
        total_loss = (self.pos_weight * pos_loss + 
                     self.angle_weight * angle_loss + 
                     width_loss)
        
        return total_loss

# Hyperparameters
BATCH_SIZE = 16  # Increased since ResNet is more efficient
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
IOU_THRESHOLD = 0.25
MODEL_TYPE = 'resnet34'  # or 'resnet18'

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
model = create_grasp_model(MODEL_TYPE, pretrained=True).to(device)
criterion = GraspLoss(pos_weight=2.0, angle_weight=0.5)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# Tracking
train_losses = []
val_losses = []
train_success_rates = []
val_success_rates = []
lrs = []

print("Starting training...")
print(f"Model: {MODEL_TYPE}")
print(f"Dataset size: {len(full_dataset)} (Train: {train_size}, Val: {val_size})")

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    running_train_loss = 0.0
    train_success_count = 0
    train_total_count = 0
    train_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        if batch is None:
            continue
            
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device)
        pos_grasps = batch['pos_grasps']
        neg_grasps = batch['neg_grasps']

        # Forward pass
        pred_params = model(rgb, depth)
        
        # Get best target grasp parameters for each prediction
        target_params, ious = get_best_grasp_target(pred_params, pos_grasps, neg_grasps)
        
        # Calculate loss
        loss = criterion(pred_params, target_params)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        # Track metrics
        running_train_loss += loss.item()
        success_rate = calculate_grasp_success_rate(pred_params, pos_grasps, neg_grasps, IOU_THRESHOLD)
        train_success_count += success_rate * len(pos_grasps)
        train_total_count += len(pos_grasps)
        train_batches += 1

        # Print progress
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    # Calculate average training metrics
    avg_train_loss = running_train_loss / train_batches if train_batches > 0 else 0
    train_success_rate = train_success_count / train_total_count if train_total_count > 0 else 0
    train_losses.append(avg_train_loss)
    train_success_rates.append(train_success_rate)

    # Validation Loop
    model.eval()
    running_val_loss = 0.0
    val_success_count = 0
    val_total_count = 0
    val_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
                
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            pos_grasps = batch['pos_grasps']
            neg_grasps = batch['neg_grasps']

            # Forward pass
            pred_params = model(rgb, depth)
            
            # Get best target grasp parameters
            target_params, ious = get_best_grasp_target(pred_params, pos_grasps, neg_grasps)
            
            # Calculate loss
            val_loss = criterion(pred_params, target_params)
            
            # Calculate success rate
            success_rate = calculate_grasp_success_rate(pred_params, pos_grasps, neg_grasps, IOU_THRESHOLD)
            
            # Track metrics
            running_val_loss += val_loss.item()
            val_success_count += success_rate * len(pos_grasps)
            val_total_count += len(pos_grasps)
            val_batches += 1

    # Calculate average validation metrics
    avg_val_loss = running_val_loss / val_batches if val_batches > 0 else 0
    val_success_rate = val_success_count / val_total_count if val_total_count > 0 else 0
    
    val_losses.append(avg_val_loss)
    val_success_rates.append(val_success_rate)
    lrs.append(optimizer.param_groups[0]['lr'])

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"  Train Loss: {avg_train_loss:.4f}, Train Success: {train_success_rate:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}, Val Success: {val_success_rate:.4f}")
    print(f"  LR: {lrs[-1]:.8f}")
    print("-" * 60)

    scheduler.step()

# Save model
os.makedirs("saved_models", exist_ok=True)
model_path = f"saved_models/resnet_{MODEL_TYPE}_grasp.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'model_type': MODEL_TYPE,
    'final_success_rate': val_success_rates[-1],
    'best_success_rate': max(val_success_rates)
}, model_path)
print(f"✅ Model saved to {model_path}")

# Model Summary
print("\n" + "="*60)
print("MODEL SUMMARY")
print("="*60)
model_info = model.get_model_info()
for key, value in model_info.items():
    print(f"{key:25}: {value}")

try:
    summary(model, input_size=[(1, 3, 224, 224), (1, 1, 224, 224)])
except:
    print("Could not generate detailed summary")

# Plot Training Results
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# Loss curves
ax1.plot(range(1, EPOCHS + 1), train_losses, 'b-o', label='Training Loss', markersize=3)
ax1.plot(range(1, EPOCHS + 1), val_losses, 'r-s', label='Validation Loss', markersize=3)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training vs Validation Loss')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Success rate curves
ax2.plot(range(1, EPOCHS + 1), train_success_rates, 'b-o', label='Training Success Rate', markersize=3)
ax2.plot(range(1, EPOCHS + 1), val_success_rates, 'r-s', label='Validation Success Rate', markersize=3)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Success Rate')
ax2.set_title(f'Success Rate (IoU > {IOU_THRESHOLD})')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Learning rate
ax3.plot(range(1, EPOCHS + 1), lrs, 'purple', marker='o', markersize=3)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Learning Rate')
ax3.set_title('Learning Rate Schedule')
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# Final metrics
final_metrics = [
    f"Model: {MODEL_TYPE}",
    f"Final Train Loss: {train_losses[-1]:.4f}",
    f"Final Val Loss: {val_losses[-1]:.4f}",
    f"Final Train Success: {train_success_rates[-1]:.4f}",
    f"Final Val Success: {val_success_rates[-1]:.4f}",
    f"Best Val Success: {max(val_success_rates):.4f}",
    f"IoU Threshold: {IOU_THRESHOLD}",
    f"Parameters: {model_info['total_parameters']:,}",
    f"Model
