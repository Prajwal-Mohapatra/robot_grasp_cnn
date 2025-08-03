# ===================== train.py =====================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

from loader import CornellGraspDataset
from model import GraspCNN

# --- Helper Functions ---

def custom_collate(batch):
    """
    Custom collate function to filter out samples with no valid grasps.
    This prevents errors during training if a sample has no positive grasp rectangles.
    """
    batch = [item for item in batch if item is not None and item['pos_grasps'].shape[0] > 0]
    if not batch: return None
    return {
        'rgb': torch.stack([item['rgb'] for item in batch]),
        'depth': torch.stack([item['depth'] for item in batch]),
        'pos_grasps': [item['pos_grasps'] for item in batch]
    }

def get_target_vector(rect, image_size):
    """
    Converts a 4x2 grasp rectangle into a 6-element target vector, normalized by image size.
    [center_x, center_y, sin(2*angle), cos(2*angle), width, height]
    """
    center = rect.mean(axis=0)
    dx, dy = rect[1] - rect[0], rect[2] - rect[1]
    width, height = np.linalg.norm(dx), np.linalg.norm(dy)
    angle = np.arctan2(dx[1], dx[0])
    
    # Normalize all parameters by the image size
    return np.array([
        center[0] / image_size, center[1] / image_size,
        np.sin(2 * angle), np.cos(2 * angle),
        width / image_size, height / image_size
    ])

def get_closest_grasp_target(pred, grasps_batch, image_size):
    """
    For each prediction in a batch, finds the closest ground truth grasp.
    This is necessary because there can be multiple valid grasps for an object.
    The "closest" is determined by the L2 distance between the prediction vector
    and the target vectors of all ground truth grasps.
    """
    batch_targets = []
    for i in range(pred.shape[0]):
        gt_rects = grasps_batch[i].cpu().numpy()
        pred_vec_np = pred[i].detach().cpu().numpy()
        
        # Convert all ground truth rectangles to target vectors
        gt_targets = np.array([get_target_vector(g, image_size) for g in gt_rects])
        
        # Find the ground truth vector with the minimum L2 distance to the prediction
        dists = np.linalg.norm(gt_targets - pred_vec_np, axis=1)
        best_gt_target = gt_targets[dists.argmin()]
        batch_targets.append(best_gt_target)
        
    return torch.tensor(np.array(batch_targets), dtype=torch.float32, device=pred.device)

def weighted_loss(pred, target):
    """
    Calculates a weighted Mean Squared Error (MSE) loss.
    This applies higher weights to the position and angle parameters,
    which are often more critical for a successful grasp.
    """
    loss_fn = nn.MSELoss(reduction='none')
    loss = loss_fn(pred, target)
    # Weights: [x, y, sin, cos, w, h]
    weights = torch.tensor([1.5, 1.5, 2.0, 2.0, 1.0, 1.0], device=pred.device)
    return (loss * weights).mean()

# --- Main Training Script ---

def train_model():
    """Main function to orchestrate the model training process."""
    # --- Hyperparameters ---
    IMAGE_SIZE = 300
    BATCH_SIZE = 32
    INITIAL_LR = 5e-4       # Learning rate for the regressor head
    FINETUNE_LR = 5e-6        # Learning rate for the full model fine-tuning
    EPOCHS_FROZEN = 50      # Epochs to train only the regressor head
    EPOCHS_FINETUNE = 30    # Epochs to fine-tune the entire model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Datasets and Dataloaders ---
    train_dataset = CornellGraspDataset(split='train')
    val_dataset = CornellGraspDataset(split='val')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate, num_workers=4, pin_memory=True)
    
    model = GraspCNN().to(device)
    
    # --- Logging Setup ---
    log_path = "outputs/training_log_v3.csv"
    os.makedirs("outputs/saved_models", exist_ok=True)
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'phase', 'loss', 'lr'])
    
    history = {'train_loss': [], 'val_loss': []}
    
    # --- Stage 1: Train Regressor Head ---
    print("\n--- Stage 1: Training Regressor Head (Backbone Frozen) ---")
    for param in model.features.parameters():
        param.requires_grad = False
        
    optimizer = optim.AdamW(model.regressor.parameters(), lr=INITIAL_LR)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    for epoch in range(EPOCHS_FROZEN):
        run_epoch(epoch, 'train', model, train_loader, optimizer, device, history, log_writer, IMAGE_SIZE, EPOCHS_FROZEN)
        run_epoch(epoch, 'val', model, val_loader, None, device, history, log_writer, IMAGE_SIZE, EPOCHS_FROZEN, scheduler)

    # --- Stage 2: Fine-tune Full Model ---
    print("\n--- Stage 2: Fine-tuning Full Model (Unfrozen) ---")
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer = optim.AdamW(model.parameters(), lr=FINETUNE_LR)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    for epoch in range(EPOCHS_FINETUNE):
        full_epoch = EPOCHS_FROZEN + epoch
        run_epoch(full_epoch, 'train', model, train_loader, optimizer, device, history, log_writer, IMAGE_SIZE, EPOCHS_FROZEN + EPOCHS_FINETUNE)
        run_epoch(full_epoch, 'val', model, val_loader, None, device, history, log_writer, IMAGE_SIZE, EPOCHS_FROZEN + EPOCHS_FINETUNE, scheduler)

    # --- Finalization ---
    log_file.close()
    plot_history(history)
    torch.save(model.state_dict(), "outputs/saved_models/grasp_cnn_final_v3.pth")
    print("\n✅ Final model saved to outputs/saved_models/grasp_cnn_final_v3.pth")

def run_epoch(epoch, phase, model, loader, optimizer, device, history, log_writer, image_size, total_epochs, scheduler=None):
    """Runs a single epoch of training or validation."""
    is_train = phase == 'train'
    model.train() if is_train else model.eval()
    
    running_loss, total_samples = 0.0, 0
    
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs} [{phase.capitalize()}]")
    
    for batch in progress_bar:
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
        
        progress_bar.set_postfix(loss=running_loss / total_samples if total_samples > 0 else 0)
        
    epoch_loss = running_loss / total_samples if total_samples > 0 else 0
    lr = optimizer.param_groups[0]['lr'] if optimizer else 0
    history[f'{phase}_loss'].append(epoch_loss)
    log_writer.writerow([epoch + 1, phase, epoch_loss, lr])
    
    if not is_train and scheduler:
        scheduler.step(epoch_loss)

def plot_history(history):
    """Plots and saves the training and validation loss curves."""
    plt.figure(figsize=(12, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/loss_curve_final_v3.png")
    plt.show()

if __name__ == '__main__':
    train_model()
