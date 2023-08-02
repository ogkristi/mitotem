import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, out_channels, num_convs=2, kernel_size=3):
        super().__init__()
        
        layers = []
        for _ in range(num_convs):
            layers.append(nn.LazyConv2d(out_channels, kernel_size=kernel_size))
            layers.append(nn.ReLU())
        
        self.convblock = nn.Sequential(*layers)
        self.poolblock = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, X):
        skip = self.convblock(X)
        Y = self.poolblock(skip)
        return Y, skip
        
class DecoderBlock(nn.Module):
    def __init__(self, out_channels, num_convs=2, kernel_size=3):
        super().__init__()
        
        self.unpoolblock = nn.LazyConvTranspose2d(
            out_channels, kernel_size=2, stride=2, bias=False)
        
        layers = []
        for _ in range(num_convs):
            layers.append(nn.LazyConv2d(out_channels, kernel_size=kernel_size))
            layers.append(nn.ReLU())
        self.convblock = nn.Sequential(*layers)
        
    def forward(self, X, skip):
        X_up = self.unpoolblock(X)
        m, n = X_up.shape[-2:]
        d_m = (skip.shape[-2]-m) // 2
        d_n = (skip.shape[-1]-n) // 2
        # Center crop skip features to same height/width as X_up
        # then concatenate them along the output channel dimension
        X_up_cat = torch.cat((X_up,skip[:, :, d_m:m+d_m, d_n:n+d_n]), dim=1)
        Y = self.convblock(X_up_cat)
        return Y