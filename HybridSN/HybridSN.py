import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASS = 9

class HybridSN_network(nn.Module):
    def __init__(self, num_classes=9, pca_components=30, dropout=0.1):
        super(HybridSN_network, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 3, 3)),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3)),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3)),
            nn.ReLU(inplace=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True)
        )
        
        # Dummy input to calculate the size after conv4
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, pca_components, 30, 30)  # Example size, adapt if needed
            dummy_input = self.conv1(dummy_input)
            dummy_input = self.conv2(dummy_input)
            dummy_input = self.conv3(dummy_input)
            dummy_input = dummy_input.view(dummy_input.size(0), dummy_input.size(1) * dummy_input.size(4), dummy_input.size(2), dummy_input.size(3))
            dummy_input = self.conv4(dummy_input)
            flattened_size = dummy_input.numel() // dummy_input.size(0)  # Calculate the flattened size

        self.dense1 = nn.Sequential(
            nn.Linear(flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )
        
        self.dense2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )
        
        self.dense3 = nn.Sequential(
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), x.size(1) * x.size(4), x.size(2), x.size(3))
        x = self.conv4(x)
        x = x.contiguous().view(x.size(0), -1)  
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.dense3(x)
        
        return out

if __name__ == '__main__':
    model = HybridSN_network(num_classes=NUM_CLASS, pca_components=30, dropout=0.1)
    
    # Create a dummy input tensor
    dummy_input = torch.randn(1, 1, 30, 550, 400)  # Adjust the dimensions as needed
    
    # Forward pass through the network
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")
