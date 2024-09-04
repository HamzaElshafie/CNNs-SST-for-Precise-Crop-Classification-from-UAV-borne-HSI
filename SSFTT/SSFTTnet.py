import PIL
import time
import torch
import torchvision
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.nn.init as init

NUM_CLASS = 9

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

# PreNorm
class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x

class SSFTTnet(nn.Module):
    def __init__(self, in_channels=1, num_classes=9, num_tokens=2, dim=64, depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1, pca_components=None):
        super(SSFTTnet, self).__init__()
        
        self.L = num_tokens
        self.cT = dim
        self.pca_components = pca_components
        
        # If PCA is not used, apply Conv3D for spectral dimensionality reduction
        if pca_components is None:
            self.conv3d_dim_reduction = nn.Conv3d(in_channels=in_channels, out_channels=24,
                                                  kernel_size=(7, 1, 1), stride=(2, 1, 1), bias=False)
            conv_input_channels = 24  # The output of conv3d_dim_reduction will have 24 channels
        else:
            # When PCA is used, in_channels remains 1 (or the reduced number of spectral components)
            conv_input_channels = in_channels  # Typically 1 after PCA reduction
            
            # Calculate conv_output_depth based on PCA components (original logic)
            conv_output_depth = int((pca_components - 3) + 1 / 1)  # Based on kernel size (3, 3, 3)

            # Conv2D feature extraction for PCA case
            self.conv2d_features = nn.Sequential(
                nn.Conv2d(in_channels=8 * conv_output_depth, out_channels=64, kernel_size=(3, 3)),
                nn.BatchNorm2d(64),
                nn.ReLU()
            )

        # 3D feature extraction: uses the correct input channels depending on PCA usage
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(conv_input_channels, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        # Conv2D and BatchNorm layers will be initialized dynamically if PCA is not used
        self.batch_norm2d = nn.BatchNorm2d(64)
        self.relu2d = nn.ReLU()

        # Tokenization layers
        self.token_wA = nn.Parameter(torch.empty(1, self.L, 64), requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, self.cT), requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        self.pos_embedding = nn.Parameter(torch.empty(1, (num_tokens + 1), dim))
        torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer block
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()
        self.nn1 = nn.Linear(dim, num_classes)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward(self, x, mask=None):
        # If PCA is not used, apply Conv3D for dimensionality reduction
        if self.pca_components is None:
            x = self.conv3d_dim_reduction(x)  # Reduce spectral dimension with Conv3D

        # Pass through 3D feature extractor
        x = self.conv3d_features(x)

        # Calculate conv_output_depth dynamically based on depth after 3D conv (if PCA is not used)
        if self.pca_components is None:
            conv_output_depth = x.shape[2]  # Depth after Conv3D (third dimension of x)

            # Dynamically create Conv2D layer based on conv_output_depth
            self.conv2d_features = nn.Conv2d(in_channels=8 * conv_output_depth, out_channels=64, kernel_size=(3, 3))

        # Flatten the spectral and channel dimensions before 2D convolutions
        x = rearrange(x, 'b c d h w -> b (c d) h w')  # Combine channel (c) and depth (d)
        
        # Pass through 2D feature extractor (either dynamically created or pre-initialized in PCA case)
        x = self.conv2d_features(x)
        x = self.batch_norm2d(x)
        x = self.relu2d(x)

        # Tokenization
        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose
        A = torch.einsum('bij,bjk->bik', x, wa)
        A = rearrange(A, 'b h w -> b w h')  # Transpose
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)

        # Add classification token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, T), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        # Transformer
        x = self.transformer(x, mask)
        x = self.to_cls_token(x[:, 0])
        x = self.nn1(x)

        return x
  
  

if __name__ == '__main__':
    pca_components = 30  # This would be determined dynamically in practice
    model = SSFTTnet(num_classes=NUM_CLASS, pca_components=pca_components)
    model.eval()
    print(model)

    # Create a dummy input with the correct number of PCA components
    input = torch.randn(64, 1, pca_components, 13, 13)
    y = model(input)
    print(y.size())