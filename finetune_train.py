import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import AC_GRConvNet
from dataset import GraspDataset

# --- Hyperparameters ---
DATA_DIR = './data'
OUTPUT_DIR = './outputs'
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'models')
FINETUNE_EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.15
EARLY_STOPPING_PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output directories
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

def get_device():
    """Gets the appropriate device for training."""
    return DEVICE

def compute_loss(pred_maps, gt_maps):
    """
    Computes the masked loss for the generative model.
    Loss for angle and width is only computed where a grasp is present.
    """
    pred_q, pred_cos, pred_sin, pred_width = torch.split(pred_maps, 1, dim=1)
    gt_q = gt_maps['q']
    gt_cos = gt_maps['cos']
    gt_sin = gt_maps['sin']
    gt_width = gt_maps['width']

    # Loss for quality map (MSE)
    loss_q = nn.functional.mse_loss(pred_q, gt_q)
    
    # Create a mask for positive grasp regions
    mask = (gt_q > 0.5).float()
    
    # Masked loss for angle and width
    loss_cos = nn.functional.mse_loss(pred_cos * mask, gt_cos * mask)
    loss_sin = nn.functional.mse_loss(pred_sin * mask, gt_sin * mask)
    loss_width = nn.functional.mse_loss(pred_width * mask, gt_width * mask)

    # Combine losses (can be weighted if needed)
    return loss_q + loss_cos + loss_sin + loss_width

def train_one_epoch(model, device, train_loader, optimizer):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for rgbd, gt_maps in pbar:
        rgbd = rgbd.to(device)
        gt_maps = {k: v.to(device) for k, v in gt_maps.items()}

        optimizer.zero_grad()
        pred_maps = model(rgbd)
        loss = compute_loss(pred_maps, gt_maps)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
    return total_loss / len(train_loader)

def validate_one_epoch(model, device, val_loader):
    """Validates the model for one epoch."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for rgbd, gt_maps in pbar:
            rgbd = rgbd.to(device)
            gt_maps = {k: v.to(device) for k, v in gt_maps.items()}
            
            pred_maps = model(rgbd)
            loss = compute_loss(pred_maps, gt_maps)
            total_loss += loss.item()
            pbar.set_postfix({'val_loss': loss.item()})
            
    return total_loss / len(val_loader)

def main():
    """Main training function with two phases."""
    device = get_device()
    print(f"Using device: {device}")

    # Dataset and Dataloaders
    full_dataset = GraspDataset(DATA_DIR, augment=True)
    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples.")

    # Model, Optimizer, Scheduler
    model = AC_GRConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=1e-7)

        # --- PHASE 2: FINE-TUNING THE TOP LAYER ---
    
    initial_model_save_path = os.path.join(MODEL_SAVE_PATH, 'ac_grconvnet_best.pth')
    
    print("    PHASE 2: Fine-tuning Output Layer")
    
    # Load the best model from Phase 1
    print(f"Loading best model from Phase 1: {initial_model_save_path}")
    model.load_state_dict(torch.load(initial_model_save_path, map_location=device))

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze only the output_head parameters
    for param in model.output_head.parameters():
        param.requires_grad = True

    # Create a new optimizer for the unfrozen layer with a smaller learning rate
    finetune_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE / 10)
    finetune_scheduler = optim.lr_scheduler.ReduceLROnPlateau(finetune_optimizer, 'min', patience=3, factor=0.5)

    best_val_loss = float('inf') # Reset best loss for fine-tuning phase
    early_stopping_counter = 0 # Reset counter
    finetune_model_save_path = os.path.join(MODEL_SAVE_PATH, 'ac_grconvnet_finetuned.pth')

    for epoch in range(FINETUNE_EPOCHS):
        print(f"\n--- Fine-tuning Epoch {epoch+1}/{FINETUNE_EPOCHS} ---")
        train_loss = train_one_epoch(model, device, train_loader, finetune_optimizer)
        val_loss = validate_one_epoch(model, device, val_loader)
        
        #train_losses.append(train_loss) # Append to the same list for a continuous plot
        #val_losses.append(val_loss)
        
        finetune_scheduler.step(val_loss)
        current_lr = finetune_optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), finetune_model_save_path)
            print(f"✅ New best fine-tuned model saved to {finetune_model_save_path}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{EARLY_STOPPING_PATIENCE}")

        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            print("🛑 Early stopping triggered in Phase 2.")
            break
            
    # Plot and save the finetuned loss curve
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    # Add a line to show where fine-tuning started
    plt.title('Training and Validation Loss (Fine-tuning)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curve_finetuned.png'))
    plt.show()
    print("Training complete.")

if __name__ == '__main__':
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
         print(f"Error: Data directory '{DATA_DIR}' is empty or does not exist.")
         print("Please download the Cornell Grasp Dataset and place it in the 'data' folder.")
    else:
        main()

