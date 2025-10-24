import torch
import torch.optim as optim
import torch.nn as nn
# Import the new schedulers
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv  # Added for logging
from datetime import datetime  # Added for timestamping

from model import AC_GRConvNet
from dataset import GraspDataset

# --- Hyperparameters ---
DATA_DIR = './data'
OUTPUT_DIR = './outputs'
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'models')
SPLIT_DIR = os.path.join(OUTPUT_DIR, 'splits')
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, 'training_log.csv')  # Path for CSV log
BATCH_SIZE = 16
EARLY_STOPPING_PATIENCE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Training Phase Hyperparameters ---
MAIN_EPOCHS = 40
FINETUNE_EPOCHS = 20
INITIAL_LEARNING_RATE = 1e-4

# --- New Fine-Tuning Scheduler Params ---
FINETUNE_LR_MAX = 1e-5          # Peak LR for cosine anneal
FINETUNE_LR_MIN = 1e-7          # Final LR
FINETUNE_WARMUP_EPOCHS = 3      # Number of epochs to ramp up to FINETUNE_LR_MAX

# --- Data Split Ratios ---
VAL_SPLIT_RATIO = 0.25
TEST_SPLIT_RATIO = 0.15

# --- Paths for saved indices ---
TRAIN_INDICES_PATH = os.path.join(SPLIT_DIR, 'train_indices.npy')
VAL_INDICES_PATH = os.path.join(SPLIT_DIR, 'val_indices.npy')
TEST_INDICES_PATH = os.path.join(SPLIT_DIR, 'test_indices.npy')

# --- Paths for saved models ---
MODEL_SAVE_PATH_MAIN = os.path.join(MODEL_SAVE_PATH, 'ac_grconvnet_main_best.pth')
MODEL_SAVE_PATH_FINETUNE = os.path.join(MODEL_SAVE_PATH, 'ac_grconvnet_finetune_best.pth')

# Create output directories
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

def get_device():
    """Gets the appropriate device for training."""
    return DEVICE

def compute_loss(pred_maps, gt_maps):
    """
    Computes the masked loss for the generative model.
    Loss for angle and width is only computed where a grasp is present.
    """
    pred_q, pred_cos, pred_sin, pred_width = torch.split(pred_maps, 1, dim=1)
    
    # Ensure gt_maps is a dictionary and keys exist
    if not isinstance(gt_maps, dict):
        raise TypeError(f"gt_maps must be a dict, but got {type(gt_maps)}")
        
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

def print_gpu_utilization(device):
    """Prints the current GPU memory utilization if on CUDA."""
    if device.type == 'cuda':
        try:
            # torch.cuda.mem_get_info() returns (free, total)
            free_mem_b, total_mem_b = torch.cuda.mem_get_info()
            total_mem_mb = total_mem_b / (1024**2)
            used_mem_b = total_mem_b - free_mem_b
            used_mem_mb = used_mem_b / (1024**2)
            print(f"GPU Utilization: {used_mem_mb:.2f} MB / {total_mem_mb:.2f} MB ({used_mem_mb/total_mem_mb*100:.1f}%)")
        except Exception as e:
            print(f"Could not get GPU memory info: {e}")

# --- NEW CSV Logger Functions ---

def setup_logger(log_path):
    """Creates the CSV log file and writes the header if it doesn't exist."""
    file_exists = os.path.exists(log_path)
    log_file = open(log_path, 'a', newline='')
    log_writer = csv.writer(log_file)
    
    if not file_exists:
        # Write header
        headers = ['timestamp', 'epoch', 'phase', 'train_loss', 'val_loss', 'learning_rate']
        log_writer.writerow(headers)
        print(f"New log file created at {log_path}")
        
    return log_file, log_writer

def log_epoch(log_writer, epoch, phase, train_loss, val_loss, lr):
    """Logs the metrics for a single epoch to the CSV file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Format LR for consistent scientific notation
    lr_formatted = f"{lr:1.0e}" 
    log_writer.writerow([timestamp, epoch, phase, f"{train_loss:.6f}", f"{val_loss:.6f}", lr_formatted])

# --- End of Logger Functions ---

def main():
    """Main training function."""
    device = get_device()
    print(f"Using device: {device}")

    # --- Setup CSV Logger ---
    log_file, log_writer = setup_logger(LOG_FILE_PATH)

    # --- Dataset and Dataloaders ---
    try:
        full_dataset = GraspDataset(DATA_DIR, augment=True)
    except FileNotFoundError as e:
        print(e)
        log_file.close() # Close log file on error
        return
        
    dataset_size = len(full_dataset)
    
    if os.path.exists(TRAIN_INDICES_PATH) and os.path.exists(VAL_INDICES_PATH) and os.path.exists(TEST_INDICES_PATH):
        print("Loading existing data splits...")
        train_indices = np.load(TRAIN_INDICES_PATH)
        val_indices = np.load(VAL_INDICES_PATH)
        test_indices = np.load(TEST_INDICES_PATH)
    else:
        print("Creating new 60/25/15 data splits...")
        test_size = int(dataset_size * TEST_SPLIT_RATIO)
        val_size = int(dataset_size * VAL_SPLIT_RATIO)
        train_size = dataset_size - val_size - test_size
        
        indices = np.random.permutation(dataset_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]
        
        np.save(TRAIN_INDICES_PATH, train_indices)
        np.save(VAL_INDICES_PATH, val_indices)
        np.save(TEST_INDICES_PATH, test_indices)
        print(f"New splits saved to {SPLIT_DIR}")

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Dataset split: {len(train_dataset)} Train (60%), {len(val_dataset)} Val (25%), {len(test_indices)} Test (15%)")

    # --- Model, Optimizer ---
    model = AC_GRConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=1e-7)
    
    # --- STAGE 1: Main Training ---
    print(f"\n--- Starting STAGE 1: Main Training ({MAIN_EPOCHS} Epochs) ---")
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(MAIN_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{MAIN_EPOCHS + FINETUNE_EPOCHS} (Main Training) ---")
        
        train_loss = train_one_epoch(model, device, train_loader, optimizer)
        val_loss = validate_one_epoch(model, device, val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Step the scheduler based on validation loss
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr}")

        # --- Print GPU utilization ---
        print_gpu_utilization(device)

        # --- Log to CSV ---
        log_epoch(log_writer, epoch + 1, 'main', train_loss, val_loss, current_lr)

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH_MAIN)
            print(f"✅ New best main model saved to {MODEL_SAVE_PATH_MAIN}")
            early_stopping_counter = 0  # Reset counter on improvement
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{EARLY_STOPPING_PATIENCE}")

        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            print(f"🛑 Early stopping triggered in main training.")
            break
            
    print(f"--- Main Training Complete. Best model saved to {MODEL_SAVE_PATH_MAIN} ---")


    # --- STAGE 2: Fine-Tuning ---
    print(f"\n--- Starting STAGE 2: Fine-Tuning ({FINETUNE_EPOCHS} Epochs) ---")
    
    # Load the best model from stage 1
    if not os.path.exists(MODEL_SAVE_PATH_MAIN):
        print("Error: Best model from main training was not found. Aborting fine-tuning.")
        log_file.close() # Close log file on error
        return
        
    print(f"Loading best main model from {MODEL_SAVE_PATH_MAIN} for fine-tuning.")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH_MAIN))

    # Re-initialize optimizer for fine-tuning, setting the new LR
    optimizer = optim.Adam(model.parameters(), lr=FINETUNE_LR_MAX) 

    # --- Define the new schedulers for fine-tuning ---
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=FINETUNE_WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, 
                                        T_max=FINETUNE_EPOCHS - FINETUNE_WARMUP_EPOCHS, 
                                        eta_min=FINETUNE_LR_MIN)
    
    # Combine them sequentially
    #
    # *** THIS IS THE FIX ***
    # The variable is renamed from `sequential_scheduler` to `scheduler`
    # to correctly re-assign the scheduler from Stage 1.
    #
    scheduler = SequentialLR(optimizer, 
                             schedulers=[warmup_scheduler, cosine_scheduler], 
                             milestones=[FINETUNE_WARMUP_EPOCHS])
    
    best_val_loss_finetune = best_val_loss
    early_stopping_counter = 0 # Reset counter for fine-tuning
    
    for epoch in range(MAIN_EPOCHS, MAIN_EPOCHS + FINETUNE_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{MAIN_EPOCHS + FINETUNE_EPOCHS} (Fine-Tuning) ---")
        
        train_loss = train_one_epoch(model, device, train_loader, optimizer)
        val_loss = validate_one_epoch(model, device, val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # --- Step the scheduler (programmatically) ---
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} Summary: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr}")

        # --- Print GPU utilization ---
        print_gpu_utilization(device)
        
        # --- Log to CSV ---
        log_epoch(log_writer, epoch + 1, 'finetune', train_loss, val_loss, current_lr)

        # Early stopping logic for fine-tuning
        if val_loss < best_val_loss_finetune:
            best_val_loss_finetune = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH_FINETUNE)
            print(f"✅ New best fine-tuned model saved to {MODEL_SAVE_PATH_FINETUNE}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{EARLY_STOPPING_PATIENCE}")

        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            print(f"🛑 Early stopping triggered in fine-tuning.")
            break

    # --- Final Plotting ---
    print("--- Training and Fine-Tuning Complete ---")
    fig = plt.figure(figsize=(12, 6)) # Get figure handle
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    
    # Add a vertical line to show where fine-tuning started
    if len(train_losses) > MAIN_EPOCHS:
        plt.axvline(x=MAIN_EPOCHS-1, color='gray', linestyle='--', label='Fine-Tuning Start')

    plt.title('Training and Validation Loss (Main + Fine-Tuning)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curve_full.png'))
    plt.close(fig) # Close the figure to free memory
    print(f"Full training complete. Loss curve saved to {os.path.join(OUTPUT_DIR, 'loss_curve_full.png')}")
    
    # --- Close the log file ---
    log_file.close()
    print(f"Training log saved to {LOG_FILE_PATH}")

if __name__ == '__main__':
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
         print(f"Error: Data directory '{DATA_DIR}' is empty or does not exist.")
         print("Please download the Cornell Grasp Dataset and place it in the 'data' folder.")
    else:
        main()

