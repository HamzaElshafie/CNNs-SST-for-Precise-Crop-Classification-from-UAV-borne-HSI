import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

NUM_CLASS = 9

class HybridSN_network(nn.Module):
    def __init__(self, num_classes=9, pca_components=30, dropout=0.1):
        super(HybridSN_network, self).__init__()

        self.conv1 = nn.Sequential(
                    nn.Conv3d(
                    in_channels=1,
                    out_channels=8,
                    kernel_size=(7, 3, 3)),
                    nn.ReLU(inplace=True))

        conv1_output_depth = (pca_components - 7) + 1 # Calculates the output depth after the 3D conv because there is no padding
        print(f"Output dimension 1: {conv1_output_depth}")
        
        self.conv2 = nn.Sequential(
                    nn.Conv3d(
                    in_channels=8,
                    out_channels=16,
                    kernel_size=(5, 3, 3)),
                    nn.ReLU(inplace=True))
        
        conv2_output_depth = (conv1_output_depth - 5) + 1 # Calculates the output depth after the 3D conv because there is no padding
        print(f"Output dimension 2: {conv2_output_depth}")

        self.conv3 = nn.Sequential(
                    nn.Conv3d(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=(3, 3, 3)),
                    nn.ReLU(inplace=True))
        
        conv3_output_depth = (conv2_output_depth - 3) + 1 # Calculates the output depth after the 3D conv because there is no padding
        print(f"Output dimension 3: {conv3_output_depth}")

        self.conv4 = nn.Sequential(
                    nn.Conv2d(
                    in_channels=32 * conv3_output_depth,
                    out_channels=64,
                    kernel_size=(3, 3)),
                    nn.ReLU(inplace=True))
        
        self.dense1 = nn.Sequential(
                    nn.Linear(18496,256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.4))
        
        self.dense2 = nn.Sequential(
                    nn.Linear(256,128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.4))
        
        self.dense3 = nn.Sequential(
                    nn.Linear(128,num_classes)
                   )
        
    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = self.conv1(x)
        print(f"After conv1: {x.shape}")
        x = self.conv2(x)
        print(f"After conv2: {x.shape}")
        x = self.conv3(x)
        print(f"After conv3: {x.shape}")
        x = x.view(x.size(0),x.size(1)*x.size(4),x.size(2),x.size(3))
        x = self.conv4(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.dense3(x)
        
        return out
        
if __name__ == '__main__':
    # Create dummy values for img_rows, img_columns, and band_dim
    img_rows, img_columns, band_dim = 550, 400, 270
    model = HybridSN_network(num_classes=NUM_CLASS, pca_components=30, dropout=0.1)
    #summary(model, (1, img_rows, img_columns, band_dim))

