import torch
import random
import os
from dataset_loader import CornellGraspDataset
from model import GraspCNN
from visualize import show_rgb_depth_grasps

def predict():
    """
    Loads a trained GraspCNN model and visualizes predictions on random samples
    from the Cornell Grasp Dataset.
    """
    # --- Configuration ---
    MODEL_PATH = "outputs/saved_models/grasp_cnn_best.pth"
    DATASET_PATH = "./data/cornell-grasp"
    OUTPUT_DIR = "outputs/prediction"
    N_SAMPLES = 5

    # --- Setup ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Model ---
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        return

    model = GraspCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("✅ Model loaded successfully.")

    # --- Load Dataset ---
    # We use the 'val' split to predict on data the model hasn't trained on.
    dataset = CornellGraspDataset(root=DATASET_PATH, split='val')
    print(f"✅ Dataset loaded. Found {len(dataset)} samples in the validation set.")

    # --- Generate Predictions ---
    for i in range(N_SAMPLES):
        idx = random.randint(0, len(dataset) - 1)
        
        # Ensure the loaded sample is not None
        sample = dataset[idx]
        if sample is None:
            print(f"⚠️ Skipping invalid sample at index {idx}.")
            continue

        rgb = sample['rgb'].to(device)
        depth = sample['depth'].to(device)
        
        print(f"Running prediction for sample {i+1}/{N_SAMPLES}...")
        with torch.no_grad():
            # Add a batch dimension (B, C, H, W) for the model
            pred_grasp = model(rgb.unsqueeze(0), depth.unsqueeze(0))

        save_path = os.path.join(OUTPUT_DIR, f"prediction_{i+1:02d}.png")
        
        show_rgb_depth_grasps(
            rgb=rgb, 
            depth=depth, 
            grasp=sample['grasp'], 
            pred_grasp=pred_grasp[0], 
            save_path=save_path
        )
        print(f"✅ Prediction saved to: {save_path}")

if __name__ == '__main__':
    predict()
