# ===================== model_comparison.py =====================
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary
import time

# Import both models
from model import create_grasp_model  # ResNet models
# Assuming the original models are in the original model.py file
# from original_model import GraspCNN, SimpleGraspCNN

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def measure_inference_time(model, input_shape, device='cuda', num_trials=100):
    """Measure inference time"""
    model.eval()
    model.to(device)
    
    # Warm up
    with torch.no_grad():
        rgb = torch.randn(1, 3, *input_shape).to(device)
        depth = torch.randn(1, 1, *input_shape).to(device)
        for _ in range(10):
            _ = model(rgb, depth)
    
    # Measure time
    times = []
    with torch.no_grad():
        for _ in range(num_trials):
            rgb = torch.randn(1, 3, *input_shape).to(device)
            depth = torch.randn(1, 1, *input_shape).to(device)
            
            start_time = time.time()
            _ = model(rgb, depth)
            torch.cuda.synchronize() if device == 'cuda' else None
            end_time = time.time()
            
            times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)

def compare_model_architectures():
    """Compare different model architectures"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (224, 224)
    
    models = {
        'ResNet-34': create_grasp_model('resnet34', pretrained=True),
        'ResNet-18': create_grasp_model('resnet18', pretrained=True),
    }
    
    print("🔍 Model Architecture Comparison")
    print("=" * 70)
    
    results = {}
    
    for name, model in models.items():
        print(f"\n📊 {name}:")
        
        # Parameter count
        total_params, trainable_params = count_parameters(model)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        print(f"   Total Parameters: {total_params:,}")
        print(f"   Trainable Parameters: {trainable_params:,}")
        print(f"   Model Size: {model_size_mb:.2f} MB")
        
        # Inference time
        try:
            mean_time, std_time = measure_inference_time(model, input_shape, device)
            print(f"   Inference Time: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")
        except Exception as e:
            print(f"   Inference Time: Error - {e}")
            mean_time = 0
        
        results[name] = {
            'params': total_params,
            'size_mb': model_size_mb,
            'inference_time': mean_time * 1000,  # Convert to ms
            'model': model
        }
        
        # Model summary
        try:
            print(f"   Architecture Summary:")
            summary(model, input_size=[(1, 3, 224, 224), (1, 1, 224, 224)], 
                   verbose=0, col_names=["output_size", "num_params"])
        except Exception as e:
            print(f"   Could not generate summary: {e}")
    
    return results

def visualize_comparison(results):
    """Visualize model comparison results"""
    models = list(results.keys())
    params = [results[m]['params'] / 1e6 for m in models]  # Convert to millions
    sizes = [results[m]['size_mb'] for m in models]
    times = [results[m]['inference_time'] for m in models]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Parameters comparison
    bars1 = axes[0].bar(models, params, color=['#3498db', '#e74c3c'])
    axes[0].set_ylabel('Parameters (Millions)')
    axes[0].set_title('Model Parameters')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, param in zip(bars1, params):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{param:.1f}M', ha='center', va='bottom')
    
    # Model size comparison
    bars2 = axes[1].bar(models, sizes, color=['#2ecc71', '#f39c12'])
    axes[1].set_ylabel('Model Size (MB)')
    axes[1].set_title('Model Size')
    axes[1].tick_params(axis='x', rotation=45)
    
    for bar, size in zip(bars2, sizes):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{size:.1f}MB', ha='center', va='bottom')
    
    # Inference time comparison
    bars3 = axes[2].bar(models, times, color=['#9b59b6', '#1abc9c'])
    axes[2].set_ylabel('Inference Time (ms)')
    axes[2].set_title('Inference Time')
    axes[2].tick_params(axis='x', rotation=45)
    
    for bar, time_val in zip(bars3, times):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{time_val:.1f}ms', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_output_differences():
    """Analyze output differences between models"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models
    resnet34 = create_grasp_model('resnet34', pretrained=True).to(device)
    resnet18 = create_grasp_model('resnet18', pretrained=True).to(device)
    
    # Set to eval mode
    resnet34.eval()
    resnet18.eval()
    
    # Generate sample inputs
    rgb = torch.randn(5, 3, 224, 224).to(device)
    depth = torch.randn(5, 1, 224, 224).to(device)
    
    with torch.no_grad():
        output34 = resnet34(rgb, depth)
        output18 = resnet18(rgb, depth)
    
    print("\n🔍 Output Analysis:")
    print(f"ResNet-34 output shape: {output34.shape}")
    print(f"ResNet-18 output shape: {output18.shape}")
    
    # Analyze output ranges
    print(f"\nResNet-34 output ranges:")
    print(f"   x: [{output34[:, 0].min():.2f}, {output34[:, 0].max():.2f}]")
    print(f"   y: [{output34[:, 1].min():.2f}, {output34[:, 1].max():.2f}]")
    print(f"   θ: [{output34[:, 2].min():.2f}, {output34[:, 2].max():.2f}]")
    print(f"   width: [{output34[:, 3].min():.2f}, {output34[:, 3].max():.2f}]")
    
    print(f"\nResNet-18 output ranges:")
    print(f"   x: [{output18[:, 0].min():.2f}, {output18[:, 0].max():.2f}]")
    print(f"   y: [{output18[:, 1].min():.2f}, {output18[:, 1].max():.2f}]")
    print(f"   θ: [{output18[:, 2].min():.2f}, {output18[:, 2].max():.2f}]")
    print(f"   width: [{output18[:, 3].min():.2f}, {output18[:, 3].max():.2f}]")
    
    # Calculate differences
    diff = torch.abs(output34 - output18)
    print(f"\nMean absolute difference: {diff.mean():.4f}")
    print(f"Max absolute difference: {diff.max():.4f}")

def training_recommendations():
    """Provide training recommendations for each model"""
    print("\n📋 Training Recommendations:")
    print("=" * 50)
    
    print("\n🎯 ResNet-34:")
    print("   - Recommended for: High accuracy requirements")
    print("   - Batch size: 16-32 (depending on GPU memory)")
    print("   - Learning rate: 0.001 with cosine annealing")
    print("   - Training time: Longer but more stable")
    print("   - Use case: Production deployment where accuracy is critical")
    
    print("\n⚡ ResNet-18:")
    print("   - Recommended for: Fast prototyping and real-time applications")
    print("   - Batch size: 32-64")
    print("   - Learning rate: 0.001-0.002")
    print("   - Training time: Faster convergence")
    print("   - Use case: Real-time robotics applications")
    
    print("\n🔧 General Tips:")
    print("   - Use pretrained weights for faster convergence")
    print("   - Apply data augmentation for better generalization")
    print("   - Monitor validation loss to prevent overfitting")
    print("   - Use mixed precision training for faster training")
    print("   - Consider ensemble methods for critical applications")

def main():
    """Main comparison function"""
    print("🚀 Starting Model Architecture Comparison")
    
    # Compare architectures
    results = compare_model_architectures()
    
    # Visualize results
    print("\n📊 Generating comparison visualizations...")
    visualize_comparison(results)
    
    # Analyze outputs
    print("\n🔍 Analyzing output differences...")
    analyze_output_differences()
    
    # Provide recommendations
    training_recommendations()
    
    print("\n✅ Comparison complete!")
    print("📁 Results saved as 'model_comparison.png'")

if __name__ == "__main__":
    main()
