import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASS = 9

class HybridSN_network(nn.Module):
    def __init__(self, num_classes=9, spectral_bands=None, patch_size=13, dropout=0.1):
        super(HybridSN_network, self).__init__()

        self.spectral_bands = spectral_bands
        self.patch_size = patch_size

        # If spectral bands are not reduced by PCA, we apply an initial 3D Conv layer to reduce the spectral bands
        if self.spectral_bands is None:
            self.band_reduction_layer = nn.Sequential(
                nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(7, 1, 1), stride=(2, 1, 1)),  # Stride (2, 1, 1) for spectral band reduction
            )
            self.spectral_bands = 8 
        else:
            self.band_reduction_layer = None

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1 if self.band_reduction_layer is None else self.spectral_bands, out_channels=8, kernel_size=(7, 3, 3)),
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

        self.x3d_shape = self.get_shape_after_3dconv()
        print(f"Output shape after 3D convolution layers: {self.x3d_shape}")

        # 2D Convolution Layer
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.x3d_shape[1] * self.x3d_shape[2], out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True)
        )

        # Get the shape after finishing the 2D convolution layer
        self.x2d_shape = self.get_shape_after_2dconv()
        print(f"Output shape after 2D convolution layer: {self.x2d_shape}")

        # Fully Connected Layers
        self.dense1 = nn.Sequential(
            nn.Linear(self.x2d_shape, 256),
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

    def get_shape_after_2dconv(self):
        x = torch.zeros((1, self.x3d_shape[1] * self.x3d_shape[2], self.x3d_shape[3], self.x3d_shape[4]))
        with torch.no_grad():
            x = self.conv4(x)
        return x.shape[1] * x.shape[2] * x.shape[3]
    
    def get_shape_after_3dconv(self):
        x = torch.zeros((1, 1, self.spectral_bands, self.patch_size, self.patch_size))
        if self.band_reduction_layer is not None:
            x = self.band_reduction_layer(x)
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        return x.shape

    def forward(self, x):
        if self.band_reduction_layer is not None:
            x = self.band_reduction_layer(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])  # Flatten the depth dimension
        x = self.conv4(x)
        x = x.contiguous().view(x.shape[0], -1)
        x = self.dense1(x)
        x = self.dense2(x)
        out = self.dense3(x)
        return out
        

if __name__ == '__main__':
    spectral_bands = 30  # Use None if you want to apply spectral band reduction via Conv3D
    patch_size = 13

    model = HybridSN_network(num_classes=NUM_CLASS, spectral_bands=spectral_bands, patch_size=patch_size, dropout=0.1)
    model.eval()
    
    # Create a dummy input with the correct number of spectral bands
    input = torch.randn(64, 1, spectral_bands, patch_size, patch_size)
    y = model(input)
    print(y.size())
