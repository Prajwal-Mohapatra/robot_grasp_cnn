# ===================== train.py =====================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import matplotlib.pyplot as plt
import csv # For logging

from loader import CornellGraspDataset
from model import GraspCNN

# --- Helper Functions ---
def custom_collate(batch):
    batch = [item for item in batch if item is not None and item['pos_grasps'].shape[0] > 0]
    if not batch: return None
    return {'rgb': torch.stack([item['rgb'] for item in batch]),
            'depth': torch.stack([item['depth'] for item in batch]),
            'pos_grasps': [item['pos_grasps'] for item in batch]}

def get_target_vector(rect, image_size):
    center = rect.mean(axis=0)
    dx, dy = rect[1] - rect[0], rect[2] - rect[1]
    width, height = np.linalg.norm(dx), np.linalg.norm(dy)
    angle = np.arctan2(dx[1], dx[0])
    return np.array([center[0] / image_size, center[1] / image_size,
                     np.sin(2 * angle), np.cos(2 * angle),
                     width / image_size, height / image_size])

def get_closest_grasp_target(pred, grasps_batch, image_size):
    batch_targets = []
    for i in range(pred.shape[0]):
        gt_rects = grasps_batch[i].cpu().numpy()
        pred_vec_np = pred[i].detach().cpu().numpy()
        gt_targets = np.array([get_target_vector(g, image_size) for g in gt_rects])
        dists = np.linalg.norm(gt_targets - pred_vec_np, axis=1)
        best_gt_target = gt_targets[dists.argmin()]
        batch_targets.append(best_gt_target)
    return torch.tensor(np.array(batch_targets), dtype=torch.float32, device=pred.device)

def weighted_loss(pred, target):
    loss_fn = nn.MSELoss(reduction='none')
    loss = loss_fn(pred, target)
    weights = torch.tensor([1.5, 1.5, 2.0, 2.0, 1.0, 1.0], device=pred.device)
    return (loss * weights).mean()

# --- Main Training Script ---
def train_model():
    # Hyperparameters
    IMAGE_SIZE = 300
    BATCH_SIZE = 32
    INITIAL_LR = 5e-4 # Lowered initial LR for more stable training
    FINETUNE_LR = 5e-6
    EPOCHS_FROZEN = 50 # Increased epochs for more learning
    EPOCHS_FINETUNE = 30
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = CornellGraspDataset(split='train')
    val_dataset = CornellGraspDataset(split='val')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate, num_workers=2)
    
    model = GraspCNN().to(device)
    
    # Setup CSV Logger
    log_path = "outputs/training_log.csv"
    os.makedirs("outputs", exist_ok=True)
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'phase', 'loss', 'lr'])
    
    history = {'train_loss': [], 'val_loss': []}
    
    # Stage 1: Train Regressor Head
    print("--- Stage 1: Training Regressor Head ---")
    for param in model.features.parameters():
        param.requires_grad = False
    optimizer = optim.AdamW(model.regressor.parameters(), lr=INITIAL_LR) # Using AdamW
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    for epoch in range(EPOCHS_FROZEN):
        run_epoch(epoch, 'train', model, train_loader, optimizer, device, history, log_writer, IMAGE_SIZE, EPOCHS_FROZEN)
        run_epoch(epoch, 'val', model, val_loader, optimizer, device, history, log_writer, IMAGE_SIZE, EPOCHS_FROZEN, scheduler)

    # Stage 2: Fine-tune Full Model
    print("\n--- Stage 2: Fine-tuning Full Model ---")
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=FINETUNE_LR) # Using AdamW
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    for epoch in range(EPOCHS_FINETUNE):
        full_epoch = EPOCHS_FROZEN + epoch
        run_epoch(full_epoch, 'train', model, train_loader, optimizer, device, history, log_writer, IMAGE_SIZE, EPOCHS_FROZEN + EPOCHS_FINETUNE)
        run_epoch(full_epoch, 'val', model, val_loader, optimizer, device, history, log_writer, IMAGE_SIZE, EPOCHS_FROZEN + EPOCHS_FINETUNE, scheduler)

    log_file.close()
    plot_history(history)
    torch.save(model.state_dict(), "outputs/saved_models/grasp_cnn_final_v2.pth")
    print("✅ Final model saved.")

def run_epoch(epoch, phase, model, loader, optimizer, device, history, log_writer, image_size, total_epochs, scheduler=None):
    is_train = phase == 'train'
    model.train() if is_train else model.eval()
    
    running_loss, total_samples = 0.0, 0
    
    for batch in loader:
        if batch is None: continue
        rgb, depth, grasps = batch['rgb'].to(device), batch['depth'].to(device), batch['pos_grasps']
        
        with torch.set_grad_enabled(is_train):
            pred = model(rgb, depth)
            target = get_closest_grasp_target(pred, grasps, image_size)
            loss = weighted_loss(pred, target)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        running_loss += loss.item() * rgb.size(0)
        total_samples += rgb.size(0)
        
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    lr = optimizer.param_groups[0]['lr']
    history[f'{phase}_loss'].append(epoch_loss)
    log_writer.writerow([epoch + 1, phase, epoch_loss, lr])
    
    if is_train:
        print(f"Epoch {epoch + 1}/{total_epochs}: Train Loss: {epoch_loss:.6f}", end=' | ')
    else:
        print(f"Val Loss: {epoch_loss:.6f}")
        if scheduler: scheduler.step(epoch_loss)

def plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.savefig("outputs/loss_curve_final_v2.png")
    plt.show()

if __name__ == '__main__':
    os.makedirs("outputs/saved_models", exist_ok=True)
    train_model()
