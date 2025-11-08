import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
# Import new schedulers for warmup
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
from datetime import datetime
import torch.utils.data.dataloader # Import for default_collate

from model import AC_GRConvNet
from dataset import GraspDataset

# --- Hyperparameters ---
DATA_DIR = './data'
OUTPUT_DIR = './outputs'
MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, 'models')
SPLIT_DIR = os.path.join(OUTPUT_DIR, 'splits')
LOG_FILE_PATH = os.path.join(OUTPUT_DIR, 'training_log.csv')
BATCH_SIZE = 16
EARLY_STOPPING_PATIENCE = 8
FINETUNE_EARLY_STOPPING_PATIENCE = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Training Phase Hyperparameters ---
MAIN_EPOCHS = 50
FINETUNE_EPOCHS = 20
INITIAL_LEARNING_RATE = 1e-4 # Max LR after warmup
WEIGHT_DECAY = 1e-4

# --- NEW: Loss Weighting ---
ANGLE_LOSS_WEIGHT = 0.5
WIDTH_LOSS_WEIGHT = 2.0
# Q_LOSS is handled by Focal Loss, which has its own balancing

# --- NEW: LR Warmup ---
WARMUP_EPOCHS = 5
WARMUP_START_LR = 1e-6 # Start LR for warmup

# --- Fine-Tuning Params ---
FINETUNE_BATCH_SIZE = 8
FINETUNE_LR_MAX = 1e-5
FINETUNE_LR_MIN = 1e-7

# --- Data Split Ratios (75/15/10) ---
VAL_SPLIT_RATIO = 0.15   # 15%
TEST_SPLIT_RATIO = 0.10  # 10%
# (Training will be 75%)

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

# --- NEW: Safe Collate Function ---
def collate_fn_safe(batch):
    """
    A custom collate_fn that filters out None values.
    This is to handle errors during data loading (e.g., corrupt files).
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        # Return empty tensors if the entire batch was corrupt
        return torch.tensor([]), {}
    # Use the default collate function on the "clean" batch
    return torch.utils.data.dataloader.default_collate(batch)

# --- NEW: Focal Loss Implementation ---
class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for binary or (soft) probabilistic targets.
    alpha: balances positive/negative examples
    gamma: focuses on hard-to-classify examples
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: (N, C, H, W) tensor with probabilities (model output)
        targets: (N, C, H, W) tensor with soft targets (0.0 to 1.0)
        """
        # Our model's Q-map already applies sigmoid, so inputs are probs.
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # p_t = y*p + (1-y)*(1-p)
        # For soft targets, this is a bit different, but this approximation works
        p_t = targets * inputs + (1 - targets) * (1 - inputs)
        
        # Calculate focal weight
        focal_weight = (1 - p_t).pow(self.gamma)
        
        # Calculate alpha weight
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        # Final loss
        loss = alpha_t * focal_weight * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# Instantiate the loss function
focal_loss_fn = BinaryFocalLoss(alpha=0.25, gamma=2.0).to(DEVICE)

# --------------------------------

def get_device():
    """Gets the appropriate device for training."""
    return DEVICE

def compute_loss(pred_maps, gt_maps):
    """
    Computes the masked loss for the generative model.
    Uses Focal Loss for Quality and weighted MSE for others.
    
    Returns:
        tuple: (total_loss_tensor, loss_dict)
        - total_loss_tensor: The computed total loss (for backprop)
        - loss_dict: A dictionary of floats for logging (total, q, angle, width)
    """
    pred_q, pred_cos, pred_sin, pred_width = torch.split(pred_maps, 1, dim=1)
    
    if not isinstance(gt_maps, dict):
        raise TypeError(f"gt_maps must be a dict, but got {type(gt_maps)}")
        
    gt_q = gt_maps['q']
    gt_cos = gt_maps['cos']
    gt_sin = gt_maps['sin']
    gt_width = gt_maps['width']

    # --- NEW: Use Focal Loss for Quality Map ---
    loss_q = focal_loss_fn(pred_q, gt_q)
    
    # Mask for angle/width loss (only where GT quality is high)
    mask = (gt_q > 0.5).float()
    
    # --- NEW: Weighted MSE for Angle and Width ---
    loss_cos = nn.functional.mse_loss(pred_cos * mask, gt_cos * mask)
    loss_sin = nn.functional.mse_loss(pred_sin * mask, gt_sin * mask)
    loss_angle = ANGLE_LOSS_WEIGHT * (loss_cos + loss_sin)
    
    loss_width = WIDTH_LOSS_WEIGHT * nn.functional.mse_loss(pred_width * mask, gt_width * mask)

    total_loss = loss_q + loss_angle + loss_width

    # Create dictionary for logging
    loss_dict = {
        'total': total_loss.item(),
        'q': loss_q.item(),
        'angle': loss_angle.item(),
        'width': loss_width.item()
    }

    return total_loss, loss_dict

def train_one_epoch(model, device, train_loader, optimizer):
    """
    Trains the model for one epoch.
    
    Returns:
        dict: A dictionary of average losses for the epoch.
    """
    model.train()
    epoch_losses = {'total': 0.0, 'q': 0.0, 'angle': 0.0, 'width': 0.0}
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for rgbd, gt_maps in pbar:
        # --- NEW: Check for empty batch from collate_fn_safe ---
        if not rgbd.numel():
            continue # Skip this corrupt batch
        # --- End of check ---

        rgbd = rgbd.to(device)
        gt_maps = {k: v.to(device) for k, v in gt_maps.items()}

        optimizer.zero_grad()
        pred_maps = model(rgbd)
        
        # Get total loss tensor for backprop and loss dict for logging
        total_loss_tensor, loss_dict = compute_loss(pred_maps, gt_maps)
        
        total_loss_tensor.backward()
        optimizer.step()

        # Aggregate the floats from the dictionary
        for key in epoch_losses:
            epoch_losses[key] += loss_dict[key]

        # Update tqdm progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}", 
            'q': f"{loss_dict['q']:.4f}",
            'ang': f"{loss_dict['angle']:.4f}",
            'w': f"{loss_dict['width']:.4f}"
        })
        
    # Calculate average losses
    num_batches = len(pbar)
    if num_batches == 0:
        return {'total': 0.0, 'q': 0.0, 'angle': 0.0, 'width': 0.0}
        
    avg_losses = {key: val / num_batches for key, val in epoch_losses.items()}
    return avg_losses

def validate_one_epoch(model, device, val_loader):
    """
    Validates the model for one epoch.
    
    Returns:
        dict: A dictionary of average losses for the epoch.
    """
    model.eval()
    epoch_losses = {'total': 0.0, 'q': 0.0, 'angle': 0.0, 'width': 0.0}
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating", leave=False)
        for rgbd, gt_maps in pbar:
            # --- NEW: Check for empty batch from collate_fn_safe ---
            if not rgbd.numel():
                continue # Skip this corrupt batch
            # --- End of check ---
            
            rgbd = rgbd.to(device)
            gt_maps = {k: v.to(device) for k, v in gt_maps.items()}
            
            pred_maps = model(rgbd)
            
            # Get loss dict for logging (don't need the tensor)
            _total_loss_tensor, loss_dict = compute_loss(pred_maps, gt_maps)
            
            # Aggregate floats
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key]

            # Update tqdm progress bar
            pbar.set_postfix({
                'val_loss': f"{loss_dict['total']:.4f}",
                'val_q': f"{loss_dict['q']:.4f}",
                'val_ang': f"{loss_dict['angle']:.4f}",
                'val_w': f"{loss_dict['width']:.4f}"
            })
    
    # Calculate average losses
    num_batches = len(pbar)
    if num_batches == 0:
        return {'total': 0.0, 'q': 0.0, 'angle': 0.0, 'width': 0.0}
        
    avg_losses = {key: val / num_batches for key, val in epoch_losses.items()}
    return avg_losses

def print_gpu_utilization(device):
    """Prints the current GPU memory utilization if on CUDA."""
    if device.type == 'cuda':
        try:
            free_mem_b, total_mem_b = torch.cuda.mem_get_info()
            total_mem_mb = total_mem_b / (1024**2)
            used_mem_b = total_mem_b - free_mem_b
            used_mem_mb = used_mem_b / (1024**2)
            print(f"GPU Utilization: {used_mem_mb:.2f} MB / {total_mem_mb:.2f} MB ({used_mem_mb/total_mem_mb*100:.1f}%)")
        except Exception as e:
            print(f"Could not get GPU memory info: {e}")

def setup_logger(log_path):
    """Creates/overwrites the CSV log file and writes the header."""
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    # Add new headers for individual losses
    headers = [
        'timestamp', 'epoch', 'phase', 
        'train_loss_total', 'train_loss_q', 'train_loss_angle', 'train_loss_width',
        'val_loss_total', 'val_loss_q', 'val_loss_angle', 'val_loss_width',
        'learning_rate'
    ]
    log_writer.writerow(headers)
    print(f"New log file created at {log_path}")
    return log_file, log_writer

def log_epoch(log_writer, epoch, phase, train_loss_dict, val_loss_dict, lr):
    """Logs the metrics for a single epoch to the CSV file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lr_formatted = f"{lr:.1e}"
    
    # Format all values, assuming the dicts are valid
    row = [
        timestamp, epoch, phase,
        f"{train_loss_dict['total']:.6f}", f"{train_loss_dict['q']:.6f}", f"{train_loss_dict['angle']:.6f}", f"{train_loss_dict['width']:.6f}",
        f"{val_loss_dict['total']:.6f}", f"{val_loss_dict['q']:.6f}", f"{val_loss_dict['angle']:.6f}", f"{val_loss_dict['width']:.6f}",
        lr_formatted
    ]
    log_writer.writerow(row)

def set_model_requires_grad(model, requires_grad=False, layers_to_unfreeze=None):
    """
    Helper function to freeze/unfreeze model layers.
    `layers_to_unfreeze` is a list of string *prefixes*
    """
    print(f"Setting requires_grad = {requires_grad} for all parameters...")
    for name, param in model.named_parameters():
        param.requires_grad = requires_grad
    
    if layers_to_unfreeze:
        print("Unfreezing specific layers...")
        for name, param in model.named_parameters():
            for prefix in layers_to_unfreeze:
                if name.startswith(prefix):
                    param.requires_grad = True
                    # print(f"  -> Unfreezing {name}") # Uncomment for debugging
                    break
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {trainable_params:,}")


def main():
    """Main training function."""
    device = get_device()
    print(f"Using device: {device}")
    log_file, log_writer = setup_logger(LOG_FILE_PATH)

    # --- Dataset and Dataloaders ---
    try:
        train_full_dataset = GraspDataset(DATA_DIR, augment=True)
        val_full_dataset = GraspDataset(DATA_DIR, augment=False)
    except FileNotFoundError as e:
        print(e); log_file.close(); return
        
    dataset_size = len(train_full_dataset)
    if dataset_size == 0:
        print("Error: Dataset is empty. Please check the data directory and file paths.")
        log_file.close()
        return
        
    if os.path.exists(TRAIN_INDICES_PATH) and os.path.exists(VAL_INDICES_PATH) and os.path.exists(TEST_INDICES_PATH):
        print("Loading existing data splits...")
        train_indices = np.load(TRAIN_INDICES_PATH)
        val_indices = np.load(VAL_INDICES_PATH)
        test_indices = np.load(TEST_INDICES_PATH)
    else:
        print("Creating new 75/15/10 data splits...")
        test_size = int(dataset_size * TEST_SPLIT_RATIO)
        val_size = int(dataset_size * VAL_SPLIT_RATIO)
        train_size = dataset_size - val_size - test_size
        indices = np.random.permutation(dataset_size)
        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]
        np.save(TRAIN_INDICES_PATH, train_indices); np.save(VAL_INDICES_PATH, val_indices); np.save(TEST_INDICES_PATH, test_indices)
        print(f"New splits saved to {SPLIT_DIR}")

    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(val_full_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn_safe)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn_safe)
    
    print(f"Dataset split: {len(train_dataset)} Train (75%), {len(val_dataset)} Val (15%), {len(test_indices)} Test (10%)")
    print("Augmentation: Enabled for Training, Disabled for Validation.")
    print(f"Loss function: Focal Loss (Q) + {ANGLE_LOSS_WEIGHT}*MSE(Angle) + {WIDTH_LOSS_WEIGHT}*MSE(Width)")

    # --- Model, Optimizer ---
    model = AC_GRConvNet().to(device)
    
    # --- FIX 1: Optimizer starts at the MAX learning rate ---
    optimizer = optim.Adam(model.parameters(), 
                           lr=INITIAL_LEARNING_RATE,
                           weight_decay=WEIGHT_DECAY)
                           
    # --- FIX 2: Schedulers for Warmup + Plateau ---
    # LinearLR will multiply the optimizer's LR by a factor
    # starting at `start_factor` and ending at `end_factor`.
    warmup_scheduler = LinearLR(optimizer, 
                                start_factor=WARMUP_START_LR / INITIAL_LEARNING_RATE, # e.g., 1e-6 / 1e-4 = 0.01
                                end_factor=1.0, # End at 1.0 * optimizer.lr
                                total_iters=WARMUP_EPOCHS)
    
    plateau_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=1e-7)
    
    print(f"LR Schedulers: {WARMUP_EPOCHS} epoch linear warmup (from {WARMUP_START_LR} to {INITIAL_LEARNING_RATE}), then ReduceLROnPlateau.")
    
    # --- STAGE 1: Main Training ---
    print(f"\n--- Starting STAGE 1: Main Training ({MAIN_EPOCHS} Epochs) ---")
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    # These lists will store the *total* loss for plotting
    train_loss_history, val_loss_history = [], []
    main_epochs_completed = 0
    
    for epoch in range(MAIN_EPOCHS):
        main_epochs_completed += 1
        print(f"\n--- Epoch {epoch+1}/{MAIN_EPOCHS + FINETUNE_EPOCHS} (Main Training) ---")
        
        # train_loss_dict and val_loss_dict are DICTIONARIES
        train_loss_dict = train_one_epoch(model, device, train_loader, optimizer)
        val_loss_dict = validate_one_epoch(model, device, val_loader)
        
        # Store total loss for plotting
        train_loss_history.append(train_loss_dict['total'])
        val_loss_history.append(val_loss_dict['total'])
        
        # --- Manual Scheduler Stepping ---
        if epoch < WARMUP_EPOCHS:
            pass # LR for this epoch is already set
        elif epoch == WARMUP_EPOCHS:
             print("Warmup complete, switching to ReduceLROnPlateau scheduler.")
             plateau_scheduler.step(val_loss_dict['total']) # Initial step for plateau
        else:
            plateau_scheduler.step(val_loss_dict['total']) # Step plateau scheduler after warmup
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print the detailed losses
        print(f"Epoch {epoch+1} Summary: "
              f"Train[Total: {train_loss_dict['total']:.4f}, Q: {train_loss_dict['q']:.4f}, Ang: {train_loss_dict['angle']:.4f}, W: {train_loss_dict['width']:.4f}] | "
              f"Val[Total: {val_loss_dict['total']:.4f}, Q: {val_loss_dict['q']:.4f}, Ang: {val_loss_dict['angle']:.4f}, W: {val_loss_dict['width']:.4f}] | "
              f"LR: {current_lr:.1e}")
        
        print_gpu_utilization(device)
        
        # Log the full dictionaries
        log_epoch(log_writer, epoch + 1, 'main', train_loss_dict, val_loss_dict, current_lr)

        # Step warmup scheduler *after* logging the current LR
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()

        # Check for improvement using the *total* validation loss
        current_val_loss = val_loss_dict['total']
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH_MAIN)
            print(f"✅ New best main model saved to {MODEL_SAVE_PATH_MAIN}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{EARLY_STOPPING_PATIENCE}")

        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            print(f"🛑 Early stopping triggered in main training.")
            break
            
    print(f"--- Main Training Complete. Best model saved to {MODEL_SAVE_PATH_MAIN} ---")


    # --- STAGE 2: Fine-Tuning ---
    print(f"\n--- Starting STAGE 2: Fine-Tuning ({FINETUNE_EPOCHS} Epochs) ---")
    print(f"Creating new DataLoaders for fine-tuning with Batch Size = {FINETUNE_BATCH_SIZE}")
    finetune_train_loader = DataLoader(train_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn_safe)
    finetune_val_loader = DataLoader(val_dataset, batch_size=FINETUNE_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn_safe)

    if not os.path.exists(MODEL_SAVE_PATH_MAIN):
        print("Error: Best model from main training was not found. Aborting fine-tuning."); log_file.close(); return
        
    print(f"Loading best main model from {MODEL_SAVE_PATH_MAIN} for fine-tuning.")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH_MAIN))

    # --- NEW: Fine-Tuning Strategy: Freeze Encoder, Train Decoder + Backbone ---
    layers_to_unfreeze = [
        'acm_fusion',     # Fusion layer
        'backbone_res1',  # Backbone
        'backbone_res2',  # Backbone
        'decoder_res1',   # All decoder blocks
        'decoder_up1_conv', 'decoder_up1_bn',
        'decoder_res2',
        'decoder_up2_conv', 'decoder_up2_bn',
        'decoder_res3',
        'decoder_up3_conv', 'decoder_up3_bn',
        'output_head'     # Final output layer
    ]
    set_model_requires_grad(model, requires_grad=False, layers_to_unfreeze=layers_to_unfreeze)
    # --------------------------------------------------------------------

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), # Only pass trainable params
        lr=FINETUNE_LR_MAX,
        weight_decay=WEIGHT_DECAY
    ) 

    scheduler = CosineAnnealingLR(optimizer, 
                                  T_max=FINETUNE_EPOCHS, 
                                  eta_min=FINETUNE_LR_MIN)
    
    best_val_loss_finetune = best_val_loss
    early_stopping_counter = 0
    finetune_epochs_completed = 0
    
    for epoch in range(MAIN_EPOCHS, MAIN_EPOCHS + FINETUNE_EPOCHS):
        finetune_epochs_completed += 1
        print(f"\n--- Epoch {epoch+1}/{MAIN_EPOCHS + FINETUNE_EPOCHS} (Fine-Tuning) ---")
        
        # train_loss_dict and val_loss_dict are DICTIONARIES
        train_loss_dict = train_one_epoch(model, device, finetune_train_loader, optimizer)
        val_loss_dict = validate_one_epoch(model, device, finetune_val_loader)
        
        # Store total loss for plotting
        train_loss_history.append(train_loss_dict['total'])
        val_loss_history.append(val_loss_dict['total'])
        
        # Step fine-tuning scheduler
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print detailed losses
        print(f"Epoch {epoch+1} Summary: "
              f"Train[Total: {train_loss_dict['total']:.4f}, Q: {train_loss_dict['q']:.4f}, Ang: {train_loss_dict['angle']:.4f}, W: {train_loss_dict['width']:.4f}] | "
              f"Val[Total: {val_loss_dict['total']:.4f}, Q: {val_loss_dict['q']:.4f}, Ang: {val_loss_dict['angle']:.4f}, W: {val_loss_dict['width']:.4f}] | "
              f"LR: {current_lr:.1e}")
              
        print_gpu_utilization(device)
        
        # Log the full dictionaries
        log_epoch(log_writer, epoch + 1, 'finetune', train_loss_dict, val_loss_dict, current_lr)

        # Check for improvement
        current_val_loss = val_loss_dict['total']
        if current_val_loss < best_val_loss_finetune:
            best_val_loss_finetune = current_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH_FINETUNE)
            print(f"✅ New best fine-tuned model saved to {MODEL_SAVE_PATH_FINETUNE}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{FINETUNE_EARLY_STOPPING_PATIENCE}")

        if early_stopping_counter >= FINETUNE_EARLY_STOPPING_PATIENCE:
            print(f"🛑 Early stopping triggered in fine-tuning.")
            break

    # --- Final Plotting ---
    print("--- Training and Fine-Tuning Complete ---")
    fig = plt.figure(figsize=(12, 6))
    
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    
    # Add vertical line if fine-tuning actually ran
    if finetune_epochs_completed > 0:
        plt.axvline(x=main_epochs_completed - 1, color='gray', linestyle='--', label='Fine-Tuning Start')

    plt.title('Training and Validation Loss (Main + Fine-Tuning)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curve_full.png'))
    plt.close(fig)
    print(f"Full training complete. Loss curve saved to {os.path.join(OUTPUT_DIR, 'loss_curve_full.png')}")
    
    log_file.close()
    print(f"Training log saved to {LOG_FILE_PATH}")

if __name__ == '__main__':
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
         print(f"Error: Data directory '{DATA_DIR}' is empty or does not exist.")
         print("Please download the Cornell Grasp Dataset and place it in the 'data' folder.")
    else:
        main()
