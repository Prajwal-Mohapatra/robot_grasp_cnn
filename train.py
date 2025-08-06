import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import os
import csv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from loader import CornellGraspDataset
from model import GraspCNN

def custom_collate(batch):
    """
    Custom collate function to handle lists of varying-size tensors.
    This is necessary because each image can have a different number of
    ground truth bounding boxes ('gt_bbs').
    """
    # Filter out None items if any worker failed to load a sample
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # Regular tensors that can be stacked
    rgb_tensors = torch.stack([item['rgb'] for item in batch])
    depth_tensors = torch.stack([item['depth'] for item in batch])
    quality_maps = torch.stack([item['quality_map'] for item in batch])
    angle_maps = torch.stack([item['angle_map'] for item in batch])
    width_maps = torch.stack([item['width_map'] for item in batch])

    # The list of bounding boxes, which cannot be stacked
    gt_bbs = [item['gt_bbs'] for item in batch]

    return {
        'rgb': rgb_tensors,
        'depth': depth_tensors,
        'quality_map': quality_maps,
        'angle_map': angle_maps,
        'width_map': width_maps,
        'gt_bbs': gt_bbs
    }


def loss_fn(pred_quality, pred_angle, pred_width, gt_quality, gt_angle, gt_width):
    """
    Calculates the combined loss for the grasp maps.
    """
    gt_size = gt_quality.shape[-2:]
    pred_quality = F.interpolate(pred_quality, size=gt_size, mode='bilinear', align_corners=True)
    pred_angle = F.interpolate(pred_angle, size=gt_size, mode='bilinear', align_corners=True)
    pred_width = F.interpolate(pred_width, size=gt_size, mode='bilinear', align_corners=True)

    quality_loss = F.binary_cross_entropy(pred_quality.squeeze(1), gt_quality)
    
    mask = gt_quality > 0.5
    
    angle_mask = mask.unsqueeze(1).expand_as(pred_angle)
    
    angle_loss = F.mse_loss(pred_angle[angle_mask], gt_angle[angle_mask]) if mask.any() else torch.tensor(0.0, device=pred_quality.device)
    
    width_loss = F.mse_loss(pred_width.squeeze(1)[mask], gt_width[mask]) if mask.any() else torch.tensor(0.0, device=pred_quality.device)
    
    return quality_loss + angle_loss + width_loss

def train_model():
    """Main function to orchestrate the model training process."""
    # --- Hyperparameters ---
    BATCH_SIZE = 16
    LR = 1e-4
    EPOCHS = 150 
    WEIGHT_DECAY = 1e-5 
    EARLY_STOP_PATIENCE = 15 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_dataset = CornellGraspDataset(split='train')
    val_dataset = CornellGraspDataset(split='val')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate)
    
    model = GraspCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/saved_models", exist_ok=True)
    
    # --- ADDED: CSV Logger Setup ---
    log_path = "outputs/training_log_v6.csv"
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'phase', 'loss', 'lr'])
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    # --- ADDED: History dictionary for plotting ---
    history = {'train_loss': [], 'val_loss': []}

    print("\n--- Training Model with Dense Prediction (v6) ---")
    for epoch in range(EPOCHS):
        train_loss = run_epoch(epoch, 'train', model, train_loader, optimizer, device, EPOCHS, log_writer)
        val_loss = run_epoch(epoch, 'val', model, val_loader, None, device, EPOCHS, log_writer, scheduler)
        
        # ADDED: Store losses for plotting
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "outputs/saved_models/grasp_cnn_best.pth")
            print(f"✨ New best model saved with validation loss: {val_loss:.6f}")
            early_stop_counter = 0 
        else:
            early_stop_counter += 1
            print(f"😟 Validation loss did not improve. Counter: {early_stop_counter}/{EARLY_STOP_PATIENCE}")

        if early_stop_counter >= EARLY_STOP_PATIENCE:
            print(f"🛑 Early stopping triggered after {EARLY_STOP_PATIENCE} epochs of no improvement.")
            break

    # --- Finalization ---
    log_file.close()
    torch.save(model.state_dict(), "outputs/saved_models/grasp_cnn_final_v6.pth")
    print("\n✅ Final model (from last epoch) saved to outputs/saved_models/grasp_cnn_final_v6.pth")
    print(f"🏆 Best model was saved with validation loss: {best_val_loss:.6f}")
    # ADDED: Call to plot the loss curve
    plot_history(history)

def run_epoch(epoch, phase, model, loader, optimizer, device, total_epochs, log_writer=None, scheduler=None):
    is_train = phase == 'train'
    model.train() if is_train else model.eval()
    running_loss = 0.0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [{phase.capitalize()}]")
    
    for batch in progress_bar:
        if batch is None: continue
        rgb, depth = batch['rgb'].to(device), batch['depth'].to(device)
        gt_quality, gt_angle, gt_width = batch['quality_map'].to(device), batch['angle_map'].to(device), batch['width_map'].to(device)
        with torch.set_grad_enabled(is_train):
            pred_quality, pred_angle, pred_width = model(rgb, depth)
            loss = loss_fn(pred_quality, pred_angle, pred_width, gt_quality, gt_angle, gt_width)
            if is_train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / len(loader))
        
    epoch_loss = running_loss / len(loader) if len(loader) > 0 else 0
    
    # ADDED: Logging to CSV file
    if log_writer:
        lr = optimizer.param_groups[0]['lr'] if optimizer else 0
        log_writer.writerow([epoch + 1, phase, epoch_loss, lr])

    if not is_train and scheduler:
        scheduler.step(epoch_loss)
        
    return epoch_loss

def plot_history(history):
    """Plots and saves the training and validation loss curves."""
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.savefig("outputs/loss_curve_v6.png")
    print("✅ Loss curve saved to outputs/loss_curve_v6.png")
    plt.close()

if __name__ == '__main__':
    train_model()
