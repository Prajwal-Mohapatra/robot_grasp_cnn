# ===================== evaluate.py =====================
import torch
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
# from dataset_loader import CornellGraspDataset
from loader import CornellGraspDataset
from model import GraspCNN
import os
import json
from torch.utils.data import DataLoader

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
    except Exception as e:
        print(f"Warning: IoU calculation failed: {e}")
        return 0.0

def calculate_distance_metrics(pred_grasp, pos_grasps):
    """Calculate various distance metrics between prediction and ground truth"""
    pred_center = pred_grasp.mean(axis=0)
    
    min_center_dist = float('inf')
    min_corner_dist = float('inf')
    
    for pos_grasp in pos_grasps:
        # Center distance
        pos_center = pos_grasp.mean(axis=0)
        center_dist = np.linalg.norm(pred_center - pos_center)
        min_center_dist = min(min_center_dist, center_dist)
        
        # Minimum corner distance
        for pred_corner in pred_grasp:
            for pos_corner in pos_grasp:
                corner_dist = np.linalg.norm(pred_corner - pos_corner)
                min_corner_dist = min(min_corner_dist, corner_dist)
    
    return {
        'center_distance': min_center_dist if min_center_dist != float('inf') else 0.0,
        'corner_distance': min_corner_dist if min_corner_dist != float('inf') else 0.0
    }

def custom_collate(batch):
    """Custom collate function for handling None samples"""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    
    return {
        'rgb': torch.stack([item['rgb'] for item in batch]),
        'depth': torch.stack([item['depth'] for item in batch]),
        'pos_grasps': [item['pos_grasps'] for item in batch],
        'neg_grasps': [item['neg_grasps'] for item in batch]
    }

def evaluate_model_comprehensive(model, dataset, device, save_dir="evaluation_results", batch_size=8):
    """
    Comprehensive evaluation of the grasp prediction model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    # Create data loader for batch processing
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    
    # Metrics storage
    all_ious = []
    all_center_distances = []
    all_corner_distances = []
    
    # Different IoU thresholds for analysis
    iou_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    success_counts = {thresh: 0 for thresh in iou_thresholds}
    
    total_samples = 0
    processed_batches = 0
    
    print(f"Evaluating model on dataset with {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch is None:
                continue
            
            # Get model predictions
            rgb_batch = batch['rgb'].to(device)
            depth_batch = batch['depth'].to(device)
            predictions = model(rgb_batch, depth_batch)
            
            # Process each sample in the batch
            for i in range(len(batch['pos_grasps'])):
                pos_grasps = batch['pos_grasps'][i]
                
                # Skip if no positive grasps
                if len(pos_grasps) == 0:
                    continue
                
                pred_grasp = predictions[i].cpu().numpy()
                pos_grasps_np = [g.cpu().numpy() for g in pos_grasps]
                
                # Calculate IoU with all positive grasps
                max_iou = 0.0
                for pos_grasp in pos_grasps_np:
                    iou = calculate_iou(pred_grasp, pos_grasp)
                    max_iou = max(max_iou, iou)
                
                all_ious.append(max_iou)
                
                # Calculate distance metrics
                distance_metrics = calculate_distance_metrics(pred_grasp, pos_grasps_np)
                all_center_distances.append(distance_metrics['center_distance'])
                all_corner_distances.append(distance_metrics['corner_distance'])
                
                # Calculate success for different thresholds
                for thresh in iou_thresholds:
                    if max_iou >= thresh:
                        success_counts[thresh] += 1
                
                total_samples += 1
            
            processed_batches += 1
            if processed_batches % 10 == 0:
                print(f"Processed {processed_batches} batches, {total_samples} samples...")
    
    # Convert to numpy arrays
    all_ious = np.array(all_ious)
    all_center_distances = np.array(all_center_distances)
    all_corner_distances = np.array(all_corner_distances)
    
    # Calculate final success rates
    success_rates = {}
    for thresh in iou_thresholds:
        success_rates[thresh] = success_counts[thresh] / total_samples if total_samples > 0 else 0.0
    
    # Calculate statistics
    iou_stats = {
        'mean': float(all_ious.mean()) if len(all_ious) > 0 else 0.0,
        'std': float(all_ious.std()) if len(all_ious) > 0 else 0.0,
        'median': float(np.median(all_ious)) if len(all_ious) > 0 else 0.0,
        'min': float(all_ious.min()) if len(all_ious) > 0 else 0.0,
        'max': float(all_ious.max()) if len(all_ious) > 0 else 0.0,
        'percentiles': {
            '25th': float(np.percentile(all_ious, 25)) if len(all_ious) > 0 else 0.0,
            '75th': float(np.percentile(all_ious, 75)) if len(all_ious) > 0 else 0.0,
            '90th': float(np.percentile(all_ious, 90)) if len(all_ious) > 0 else 0.0,
            '95th': float(np.percentile(all_ious, 95)) if len(all_ious) > 0 else 0.0
        }
    }
    
    distance_stats = {
        'center_distance': {
            'mean': float(all_center_distances.mean()) if len(all_center_distances) > 0 else 0.0,
            'std': float(all_center_distances.std()) if len(all_center_distances) > 0 else 0.0,
            'median': float(np.median(all_center_distances)) if len(all_center_distances) > 0 else 0.0
        },
        'corner_distance': {
            'mean': float(all_corner_distances.mean()) if len(all_corner_distances) > 0 else 0.0,
            'std': float(all_corner_distances.std()) if len(all_corner_distances) > 0 else 0.0,
            'median': float(np.median(all_corner_distances)) if len(all_corner_distances) > 0 else 0.0
        }
    }
    
    # Compile results
    results = {
        'total_samples': total_samples,
        'iou_statistics': iou_stats,
        'distance_statistics': distance_stats,
        'success_rates': success_rates,
        'raw_data': {
            'ious': all_ious.tolist(),
            'center_distances': all_center_distances.tolist(),
            'corner_distances': all_corner_distances.tolist()
        }
    }
    
    # Save detailed results
    results_path = os.path.join(save_dir, "comprehensive_evaluation.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Evaluation results saved to: {results_path}")
    
    # Create visualizations
    create_evaluation_plots(results, save_dir)
    
    # Print summary
    print_evaluation_summary(results)
    
    return results

def create_evaluation_plots(results, save_dir):
    """Create comprehensive evaluation plots"""
    
    # Set up the plotting style
    plt.style.use('default')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comprehensive Model Evaluation', fontsize=16, fontweight='bold')
    
    # Get data
    ious = np.array(results['raw_data']['ious'])
    center_dists = np.array(results['raw_data']['center_distances'])
    corner_dists = np.array(results['raw_data']['corner_distances'])
    
    # Handle empty data
    if len(ious) == 0:
        print("⚠️ No data available for plotting")
        return
    
    # 1. IoU Distribution
    axes[0, 0].hist(ious, bins=min(50, len(ious)//2), alpha=0.7, edgecolor='black', color='skyblue')
    axes[0, 0].axvline(ious.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ious.mean():.3f}')
    axes[0, 0].axvline(np.median(ious), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(ious):.3f}')
    axes[0, 0].set_xlabel('IoU Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('IoU Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Success Rate vs IoU Threshold
    thresholds = list(results['success_rates'].keys())
    success_rates = [results['success_rates'][t] for t in thresholds]
    
    axes[0, 1].plot(thresholds, success_rates, 'o-', linewidth=2, markersize=6, color='darkgreen')
    axes[0, 1].set_xlabel('IoU Threshold')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].set_title('Success Rate vs IoU Threshold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    # Add annotations
    for i, (thresh, rate) in enumerate(zip(thresholds, success_rates)):
        axes[0, 1].annotate(f'{rate:.3f}', (thresh, rate), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=9)
    
    # 3. Distance Metrics
    axes[0, 2].hist(center_dists, bins=min(30, len(center_dists)//2), alpha=0.7, 
                    label='Center Distance', edgecolor='black', color='orange')
    axes[0, 2].hist(corner_dists, bins=min(30, len(corner_dists)//2), alpha=0.7, 
                    label='Corner Distance', edgecolor='black', color='purple')
    axes[0, 2].set_xlabel('Distance (pixels)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Distance Metrics Distribution')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Box Plot of IoU Scores
    bp = axes[1, 0].boxplot(ious, labels=['IoU Scores'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    axes[1, 0].set_ylabel('IoU Score')
    axes[1, 0].set_title('IoU Score Distribution (Box Plot)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Cumulative Distribution of IoU
    sorted_ious = np.sort(ious)
    cumulative = np.arange(1, len(sorted_ious) + 1) / len(sorted_ious)
    
    axes[1, 1].plot(sorted_ious, cumulative, linewidth=2, color='navy')
    axes[1, 1].set_xlabel('IoU Score')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Cumulative Distribution of IoU Scores')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add threshold lines
    for thresh in [0.25, 0.50, 0.75]:
        if thresh <= ious.max():
            axes[1, 1].axvline(thresh, color='red', linestyle='--', alpha=0.5)
            axes[1, 1].text(thresh, 0.1, f'{thresh}', rotation=90, ha='right', fontsize=9)
    
    # 6. Performance Summary Table
    axes[1, 2].axis('off')
    
    # Create summary table
    summary_data = [
        ['Metric', 'Value'],
        ['Total Samples', f"{results['total_samples']}"],
        ['Mean IoU', f"{results['iou_statistics']['mean']:.4f}"],
        ['Median IoU', f"{results['iou_statistics']['median']:.4f}"],
        ['IoU Std', f"{results['iou_statistics']['std']:.4f}"],
        ['Success Rate (0.25)', f"{results['success_rates'][0.25]:.4f}"],
        ['Success Rate (0.30)', f"{results['success_rates'][0.30]:.4f}"],
        ['90th Percentile IoU', f"{results['iou_statistics']['percentiles']['90th']:.4f}"],
        ['Mean Center Dist', f"{results['distance_statistics']['center_distance']['mean']:.2f}"],
        ['Mean Corner Dist', f"{results['distance_statistics']['corner_distance']['mean']:.2f}"]
    ]
    
    table = axes[1, 2].table(cellText=summary_data[1:], colLabels=summary_data[0], 
                            cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(summary_data)):
        for j in range(len(summary_data[0])):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#40466e')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                table[(i, j)].set_facecolor('#f1f1f2')
    
    axes[1, 2].set_title('Performance Summary', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, "comprehensive_evaluation_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Evaluation plots saved to: {plot_path}")

def print_evaluation_summary(results):
    """Print a comprehensive evaluation summary"""
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("="*70)
    
    print(f"📊 Total Samples Evaluated: {results['total_samples']}")
    print(f"📈 Mean IoU: {results['iou_statistics']['mean']:.4f}")
    print(f"📉 IoU Standard Deviation: {results['iou_statistics']['std']:.4f}")
    print(f"🎯 Median IoU: {results['iou_statistics']['median']:.4f}")
    print(f"📏 IoU Range: [{results['iou_statistics']['min']:.4f}, {results['iou_statistics']['max']:.4f}]")
    
    print(f"\n📊 IoU Percentiles:")
    print(f"   25th: {results['iou_statistics']['percentiles']['25th']:.4f}")
    print(f"   75th: {results['iou_statistics']['percentiles']['75th']:.4f}")
    print(f"   90th: {results['iou_statistics']['percentiles']['90th']:.4f}")
    print(f"   95th: {results['iou_statistics']['percentiles']['95th']:.4f}")
    
    print(f"\n🎯 Success Rates at Different IoU Thresholds:")
    for thresh, rate in results['success_rates'].items():
        print(f"   IoU > {thresh}: {rate:.4f} ({rate*100:.2f}%)")
    
    print(f"\n📏 Distance Metrics:")
    print(f"   Mean Center Distance: {results['distance_statistics']['center_distance']['mean']:.2f} pixels")
    print(f"   Mean Corner Distance: {results['distance_statistics']['corner_distance']['mean']:.2f} pixels")
    
    # Performance assessment
    primary_success_rate = results['success_rates'][0.25]
    mean_iou = results['iou_statistics']['mean']
    
    print(f"\n🔍 Performance Assessment:")
    if primary_success_rate > 0.6:
        print(f"   ✅ GOOD: Success rate of {primary_success_rate:.3f} is above 60%")
    elif primary_success_rate > 0.4:
        print(f"   ⚠️  FAIR: Success rate of {primary_success_rate:.3f} is moderate (40-60%)")
    else:
        print(f"   ❌ POOR: Success rate of {primary_success_rate:.3f} is below 40%")
    
    if mean_iou > 0.3:
        print(f"   ✅ GOOD: Mean IoU of {mean_iou:.3f} is above 0.3")
    elif mean_iou > 0.2:
        print(f"   ⚠️  FAIR: Mean IoU of {mean_iou:.3f} is moderate (0.2-0.3)")
    else:
        print(f"   ❌ POOR: Mean IoU of {mean_iou:.3f} is below 0.2")
    
    print("="*70)

def evaluate_model_on_dataset(model_path, dataset_path, device=None, save_dir="evaluation_results"):
    """
    Main function to evaluate a trained model on a dataset
    """
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load model
    model = GraspCNN().to(device)
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            print(f"✅ Model loaded from: {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    else:
        print(f"❌ Model file not found: {model_path}")
        return None
    
    # Load dataset
    try:
        dataset = CornellGraspDataset(root=dataset_path, split='val')
        if len(dataset) == 0:
            print("⚠️ No validation samples found. Using full dataset.")
            dataset = CornellGraspDataset(root=dataset_path, split='all')
        
        print(f"✅ Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None
    
    # Run evaluation
    results = evaluate_model_comprehensive(model, dataset, device, save_dir)
    
    return results

def main():
    """Main function for running evaluation"""
    # Configuration
    model_path = "saved_models/grasp_cnn.pth"
    dataset_path = "./data/cornell-grasp"
    save_dir = "evaluation_results"
    
    # Run evaluation
    results = evaluate_model_on_dataset(model_path, dataset_path, save_dir=save_dir)
    
    if results is not None:
        print("\n🎉 Evaluation completed successfully!")
        print(f"📁 Results saved to: {save_dir}")
    else:
        print("\n❌ Evaluation failed!")

if __name__ == "__main__":
    main()
