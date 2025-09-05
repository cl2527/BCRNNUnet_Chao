import torch
import torch.nn as nn

class CNN3DRegressor(nn.Module):
    def __init__(self,filters = [32, 64, 128, 256]):
        super(CNN3DRegressor, self).__init__()
        
        # Define the 3D convolutional layers with downsampling
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, filters[0], kernel_size=3, stride=2, padding=1),  # Downsample by 2
            nn.BatchNorm3d(filters[0]),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(filters[0], filters[1], kernel_size=3, stride=2, padding=1),  # Downsample by 2 again
            nn.BatchNorm3d(filters[1]),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(filters[1], filters[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(filters[2]),
            nn.ReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv3d(filters[2], filters[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(filters[3]),
            nn.ReLU()
        )

        # Final fully connected layer to produce a scalar output
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # Global pooling to reduce the feature map to a single element per channel
            nn.Flatten(),
            nn.Linear(filters[3], 1)  # Output scalar
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x