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
                    kernel_size=(3, 3, 7)),
                    nn.ReLU(inplace=True))

        conv1_output_depth = int((pca_components - 3) + 1 / 1) # Calculates the output depth after the 3D conv because there is no padding
        
        self.conv2 = nn.Sequential(
                    nn.Conv3d(
                    in_channels=8,
                    out_channels=16,
                    kernel_size=(3, 3, 5)),
                    nn.ReLU(inplace=True))
        
        conv2_output_depth = int((conv1_output_depth - 3) + 1 / 1) # Calculates the output depth after the 3D conv because there is no padding
        
        self.conv3 = nn.Sequential(
                    nn.Conv3d(
                    in_channels=16,
                    out_channels=32,
                    kernel_size=(3, 3, 3)),
                    nn.ReLU(inplace=True))
        
        conv3_output_depth = int((conv2_output_depth - 3) + 1 / 1) # Calculates the output depth after the 3D conv because there is no padding
        
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
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
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
    pca_components = 30
    model = HybridSN_network(num_classes=NUM_CLASS, pca_components=pca_components)
    summary(model, (1, img_rows, img_columns, band_dim))

