# ===================== model.py =====================

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraspCNN(nn.Module):
    def __init__(self, input_size=(224, 224)):
        super(GraspCNN, self).__init__()
        
        # RGB feature extractor (3 channels)
        self.rgb_features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224x224 -> 112x112
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112 -> 56x56
            nn.Dropout2d(0.1),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56 -> 28x28
            nn.Dropout2d(0.2),
        )
        
        # Depth feature extractor (1 channel)
        self.depth_features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224x224 -> 112x112
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112 -> 56x56
            nn.Dropout2d(0.1),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56 -> 28x28
            nn.Dropout2d(0.2),
        )
        
        # Fusion layer (128 + 128 = 256 channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            nn.Dropout2d(0.3),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
            nn.Dropout2d(0.3),
        )
        
        # Calculate flattened feature size
        # After all pooling operations: 224 -> 112 -> 56 -> 28 -> 14 -> 7
        # Final size: 512 * 7 * 7 = 25088
        self.flattened_size = 512 * 7 * 7
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 8)  # 4 points × 2 coordinates (x, y)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, rgb, depth):
        """
        Forward pass
        Args:
            rgb: RGB image tensor [batch_size, 3, 224, 224]
            depth: Depth image tensor [batch_size, 1, 224, 224]
        Returns:
            Grasp coordinates [batch_size, 4, 2]
        """
        # Extract RGB features
        rgb_feat = self.rgb_features(rgb)  # [batch_size, 128, 28, 28]
        
        # Extract depth features
        depth_feat = self.depth_features(depth)  # [batch_size, 128, 28, 28]
        
        # Concatenate features
        fused_feat = torch.cat([rgb_feat, depth_feat], dim=1)  # [batch_size, 256, 28, 28]
        
        # Apply fusion layers
        fused_feat = self.fusion(fused_feat)  # [batch_size, 512, 7, 7]
        
        # Flatten
        x = fused_feat.view(fused_feat.size(0), -1)  # [batch_size, 25088]
        
        # Classify
        x = self.classifier(x)  # [batch_size, 8]
        
        # Reshape to grasp coordinates
        grasp_coords = x.view(-1, 4, 2)  # [batch_size, 4, 2]
        
        # Apply sigmoid to normalize coordinates to [0, 1] range
        # Then scale to image size (224x224)
        grasp_coords = torch.sigmoid(grasp_coords) * 224.0
        
        return grasp_coords

    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'input_size': (224, 224),
            'output_size': (4, 2)
        }

# Alternative simpler model if the above is too complex
class SimpleGraspCNN(nn.Module):
    def __init__(self, input_size=(224, 224)):
        super(SimpleGraspCNN, self).__init__()
        
        # Combined feature extractor for RGB+Depth (4 channels)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3),  # 224x224 -> 112x112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),  # 112x112 -> 56x56
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56 -> 28x28
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14x14 -> 7x7
            
            # Global Average Pooling
            nn.AdaptiveAvgPool2d((1, 1))  # 7x7 -> 1x1
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 8)  # 4 points × 2 coordinates
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, rgb, depth):
        """
        Forward pass
        Args:
            rgb: RGB image tensor [batch_size, 3, 224, 224]
            depth: Depth image tensor [batch_size, 1, 224, 224]
        Returns:
            Grasp coordinates [batch_size, 4, 2]
        """
        # Concatenate RGB and depth
        x = torch.cat([rgb, depth], dim=1)  # [batch_size, 4, 224, 224]
        
        # Extract features
        x = self.features(x)  # [batch_size, 512, 1, 1]
        
        # Flatten
        x = x.view(x.size(0), -1)  # [batch_size, 512]
        
        # Classify
        x = self.classifier(x)  # [batch_size, 8]
        
        # Reshape to grasp coordinates
        grasp_coords = x.view(-1, 4, 2)  # [batch_size, 4, 2]
        
        # Apply sigmoid to normalize coordinates to [0, 1] range
        # Then scale to image size (224x224)
        grasp_coords = torch.sigmoid(grasp_coords) * 224.0
        
        return grasp_coords

# You can use either GraspCNN or SimpleGraspCNN
# The SimpleGraspCNN is more efficient and might be easier to train
