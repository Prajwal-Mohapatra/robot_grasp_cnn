# ===================== grasp_visualization.py =====================
import torch
import random
from dataset_loader import CornellGraspDataset
from model import GraspCNN
from visualize import show_rgb_depth_grasps, show_batch_predictions, evaluate_model_performance
import os

def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = GraspCNN().to(device)
    model_path = "saved_models/grasp_cnn.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"✅ Model loaded from: {model_path}")
    else:
        print(f"❌ Model file not found: {model_path}")
        print("Please train the model first using train.py")
        return

    # Load dataset
    print("Loading dataset...")
    val_dataset = CornellGraspDataset(root="./data/cornell-grasp", split='val')
    
    if len(val_dataset) == 0:
        print("❌ No validation samples found. Using full dataset.")
        val_dataset = CornellGraspDataset(root="./data/cornell-grasp", split='all')
    
    print(f"Dataset loaded with {len(val_dataset)} samples")

    # Create output directory
    output_dir = "grasp_outputs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Option 1: Show individual predictions
    print("\n" + "="*60)
    print("INDIVIDUAL PREDICTIONS")
    print("="*60)
    
    n_individual = 5
    show_batch_predictions(model, val_dataset, device, n_samples=n_individual, save_dir=output_dir)

    # Option 2: Comprehensive evaluation
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION")
    print("="*60)
    
    n_eval = min(100, len(val_dataset))
    eval_results = evaluate_model_performance(model, val_dataset, device, n_samples=n_eval)
    
    # Save evaluation results
    import json
    eval_path = os.path.join(output_dir, "evaluation_results.json")
    
    # Convert numpy arrays to lists for JSON serialization
    eval_results_json = {
        'success_rate': float(eval_results['success_rate']),
        'avg_iou': float(eval_results['avg_iou']),
        'max_iou': float(eval_results['max_iou']),
        'min_iou': float(eval_results['min_iou']),
        'iou_std': float(eval_results['iou_std']),
        'total_samples': int(eval_results['total_samples']),
        'successful_grasps': int(eval_results['successful_grasps']),
        'iou_scores': eval_results['iou_scores'].tolist(),
        'success_flags': [bool(x) for x in eval_results['success_flags']]
    }
    
    with open(eval_path, 'w') as f:
        json.dump(eval_results_json, f, indent=2)
    
    print(f"✅ Evaluation results saved to: {eval_path}")

    # Option 3: Show some challenging cases (low IoU)
    print("\n" + "="*60)
    print("CHALLENGING CASES (Low IoU)")
    print("="*60)
    
    # Find samples with low IoU for analysis
    low_iou_indices = []
    model.eval()
    
    with torch.no_grad():
        for i in range(min(50, len(val_dataset))):
            sample = val_dataset[i]
            if sample is None or len(sample['pos_grasps']) == 0:
                continue
            
            # Get prediction
            rgb_batch = sample['rgb'].unsqueeze(0).to(device)
            depth_batch = sample['depth'].unsqueeze(0).to(device)
            pred = model(rgb_batch, depth_batch)
            pred_grasp = pred[0].cpu().numpy()
            
            # Calculate max IoU
            max_iou = 0.0
            for pos_grasp in sample['pos_grasps']:
                from visualize import calculate_iou
                iou = calculate_iou(pred_grasp, pos_grasp.cpu().numpy())
                max_iou = max(max_iou, iou)
            
            if max_iou < 0.15:  # Low IoU threshold
                low_iou_indices.append((i, max_iou))
    
    # Sort by IoU (lowest first)
    low_iou_indices.sort(key=lambda x: x[1])
    
    # Show top 3 challenging cases
    n_challenging = min(3, len(low_iou_indices))
    if n_challenging > 0:
        print(f"Found {len(low_iou_indices)} challenging cases (IoU < 0.15)")
        print(f"Showing {n_challenging} most challenging cases...")
        
        for i, (idx, iou) in enumerate(low_iou_indices[:n_challenging]):
            sample = val_dataset[idx]
            
            # Get prediction
            rgb_batch = sample['rgb'].unsqueeze(0).to(device)
            depth_batch = sample['depth'].unsqueeze(0).to(device)
            pred = model(rgb_batch, depth_batch)
            
            # Prepare visualization data
            rgb_vis = sample['rgb'].clone()
            depth_vis = sample['depth'].clone()
            
            # Denormalize RGB
            rgb_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            rgb_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            rgb_vis = rgb_vis * rgb_std + rgb_mean
            rgb_vis = torch.clamp(rgb_vis, 0, 1)
            
            # Denormalize depth
            depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min() + 1e-8)
            
            # Extract grasps
            pos_grasps = sample['pos_grasps'] if len(sample['pos_grasps']) > 0 else None
            neg_grasps = sample['neg_grasps'] if len(sample['neg_grasps']) > 0 else None
            pred_grasp = pred[0]
            
            # Save path
            save_path = os.path.join(output_dir, f"challenging_case_{i+1:02d}_iou_{iou:.3f}.png")
            
            # Visualize
            show_rgb_depth_grasps(
                rgb_vis, depth_vis, pos_grasps, neg_grasps, pred_grasp,
                save_path=save_path, show_metrics=True
            )
            
            print(f"✅ Challenging case {i+1}/{n_challenging} saved (IoU: {iou:.3f})")
    else:
        print("No challenging cases found (all IoU > 0.15)")

    # Summary
    print("\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Individual predictions: {n_individual} samples")
    print(f"📈 Evaluation samples: {n_eval} samples")
    print(f"⚠️  Challenging cases: {n_challenging} samples")
    print(f"🎯 Overall success rate: {eval_results['success_rate']:.4f}")
    print(f"📏 Average IoU: {eval_results['avg_iou']:.4f}")
    print("="*60)
    
    # List all generated files
    print("\n📄 Generated files:")
    for file in sorted(os.listdir(output_dir)):
        if file.endswith(('.png', '.json')):
            print(f"   - {file}")

if __name__ == "__main__":
    main()
