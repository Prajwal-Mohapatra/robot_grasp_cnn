# ===================== model.py =====================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math

class ResNetGraspRegressor(nn.Module):
    def __init__(self, input_size=(224, 224), pretrained=True):
        super(ResNetGraspRegressor, self).__init__()
        
        # Load pretrained ResNet-34
        self.resnet = models.resnet34(pretrained=pretrained)
        
        # Modify the first convolutional layer to accept 4 channels (RGB + Depth)
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=4,  # RGB (3) + Depth (1)
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
        # Initialize the new conv1 layer properly
        if pretrained:
            # Copy weights from original RGB channels and initialize depth channel
            with torch.no_grad():
                # Copy RGB weights
                self.resnet.conv1.weight[:, :3, :, :] = original_conv1.weight
                # Initialize depth channel weights (average of RGB channels)
                self.resnet.conv1.weight[:, 3:4, :, :] = original_conv1.weight.mean(dim=1, keepdim=True)
        
        # Remove the final fully connected layer
        self.resnet.fc = nn.Identity()
        
        # Get the number of features from ResNet-34
        self.resnet_features = 512
        
        # Dense regression head for grasp parameters
        self.regression_head = nn.Sequential(
            nn.Linear(self.resnet_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, 4)  # [x, y, θ, width]
        )
        
        # Initialize regression head weights
        self._initialize_regression_head()
    
    def _initialize_regression_head(self):
        """Initialize regression head weights"""
        for m in self.regression_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, rgb, depth):
        """
        Forward pass
        Args:
            rgb: RGB image tensor [batch_size, 3, 224, 224]
            depth: Depth image tensor [batch_size, 1, 224, 224]
        Returns:
            Grasp parameters [batch_size, 4] -> [x, y, θ, width]
        """
        # Concatenate RGB and depth
        x = torch.cat([rgb, depth], dim=1)  # [batch_size, 4, 224, 224]
        
        # Extract features using ResNet-34
        features = self.resnet(x)  # [batch_size, 512]
        
        # Apply regression head
        grasp_params = self.regression_head(features)  # [batch_size, 4]
        
        # Apply appropriate activations to constrain outputs
        # x, y: sigmoid to normalize to [0, 1] then scale to image size
        # θ: tanh to normalize to [-1, 1] then scale to [-π/2, π/2]
        # width: sigmoid to normalize to [0, 1] then scale to reasonable range
        
        x_coord = torch.sigmoid(grasp_params[:, 0]) * 224.0  # x in [0, 224]
        y_coord = torch.sigmoid(grasp_params[:, 1]) * 224.0  # y in [0, 224]
        theta = torch.tanh(grasp_params[:, 2]) * (math.pi / 2)  # θ in [-π/2, π/2]
        width = torch.sigmoid(grasp_params[:, 3]) * 150.0  # width in [0, 150] pixels
        
        return torch.stack([x_coord, y_coord, theta, width], dim=1)
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'backbone': 'ResNet-34',
            'input_size': (224, 224),
            'input_channels': 4,
            'output_size': 4,
            'output_format': '[x, y, θ, width]'
        }

# Alternative: ResNet-18 for faster training
class ResNetGraspRegressorLite(nn.Module):
    def __init__(self, input_size=(224, 224), pretrained=True):
        super(ResNetGraspRegressorLite, self).__init__()
        
        # Load pretrained ResNet-18
        self.resnet = models.resnet18(pretrained=pretrained)
        
        # Modify the first convolutional layer to accept 4 channels
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
        # Initialize the new conv1 layer
        if pretrained:
            with torch.no_grad():
                self.resnet.conv1.weight[:, :3, :, :] = original_conv1.weight
                self.resnet.conv1.weight[:, 3:4, :, :] = original_conv1.weight.mean(dim=1, keepdim=True)
        
        # Remove the final fully connected layer
        self.resnet.fc = nn.Identity()
        
        # ResNet-18 has 512 features
        self.resnet_features = 512
        
        # Simpler regression head
        self.regression_head = nn.Sequential(
            nn.Linear(self.resnet_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 4)  # [x, y, θ, width]
        )
        
        self._initialize_regression_head()
    
    def _initialize_regression_head(self):
        for m in self.regression_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, rgb, depth):
        # Concatenate RGB and depth
        x = torch.cat([rgb, depth], dim=1)
        
        # Extract features
        features = self.resnet(x)
        
        # Apply regression head
        grasp_params = self.regression_head(features)
        
        # Apply activations
        x_coord = torch.sigmoid(grasp_params[:, 0]) * 224.0
        y_coord = torch.sigmoid(grasp_params[:, 1]) * 224.0
        theta = torch.tanh(grasp_params[:, 2]) * (math.pi / 2)
        width = torch.sigmoid(grasp_params[:, 3]) * 150.0
        
        return torch.stack([x_coord, y_coord, theta, width], dim=1)

# Utility functions for grasp rectangle conversion
def grasp_params_to_rectangle(grasp_params, height=30):
    """
    Convert grasp parameters to rectangle coordinates
    Args:
        grasp_params: [x, y, θ, width] tensor
        height: fixed height of grasp rectangle
    Returns:
        Rectangle coordinates [4, 2] representing 4 corners
    """
    batch_size = grasp_params.size(0)
    rectangles = []
    
    for i in range(batch_size):
        x, y, theta, width = grasp_params[i]
        
        # Half dimensions
        half_width = width / 2
        half_height = height / 2
        
        # Rectangle corners in local coordinates (centered at origin)
        corners = torch.tensor([
            [-half_width, -half_height],  # Bottom-left
            [half_width, -half_height],   # Bottom-right
            [half_width, half_height],    # Top-right
            [-half_width, half_height]    # Top-left
        ], device=grasp_params.device)
        
        # Rotation matrix
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        rotation_matrix = torch.tensor([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ], device=grasp_params.device)
        
        # Apply rotation
        rotated_corners = torch.mm(corners, rotation_matrix.t())
        
        # Translate to center position
        translated_corners = rotated_corners + torch.tensor([x, y], device=grasp_params.device)
        
        rectangles.append(translated_corners)
    
    return torch.stack(rectangles)

def rectangle_to_grasp_params(rectangles):
    """
    Convert rectangle coordinates to grasp parameters
    Args:
        rectangles: [batch_size, 4, 2] tensor representing 4 corners
    Returns:
        Grasp parameters [batch_size, 4] -> [x, y, θ, width]
    """
    batch_size = rectangles.size(0)
    grasp_params = []
    
    for i in range(batch_size):
        rect = rectangles[i]  # [4, 2]
        
        # Calculate center
        center = rect.mean(dim=0)  # [2]
        x, y = center[0], center[1]
        
        # Calculate width (distance between adjacent corners)
        width = torch.norm(rect[1] - rect[0])
        
        # Calculate angle
        # Vector from corner 0 to corner 1
        edge_vector = rect[1] - rect[0]
        theta = torch.atan2(edge_vector[1], edge_vector[0])
        
        grasp_params.append(torch.tensor([x, y, theta, width], device=rectangles.device))
    
    return torch.stack(grasp_params)

# Choose which model to use
def create_grasp_model(model_type='resnet34', **kwargs):
    """
    Factory function to create grasp model
    Args:
        model_type: 'resnet34' or 'resnet18'
    Returns:
        Model instance
    """
    if model_type == 'resnet34':
        return ResNetGraspRegressor(**kwargs)
    elif model_type == 'resnet18':
        return ResNetGraspRegressorLite(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Example usage
if __name__ == "__main__":
    # Create model
    model = create_grasp_model('resnet34', pretrained=True)
    
    # Print model info
    info = model.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test forward pass
    rgb = torch.randn(2, 3, 224, 224)
    depth = torch.randn(2, 1, 224, 224)
    
    with torch.no_grad():
        output = model(rgb, depth)
        print(f"\nOutput shape: {output.shape}")
        print(f"Output (first sample): {output[0]}")
        
        # Convert to rectangle for visualization
        rectangles = grasp_params_to_rectangle(output)
        print(f"Rectangle shape: {rectangles.shape}")
