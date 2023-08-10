import torch
from torch import nn

class UNet(nn.Module):
    def __init__(self, channels_out=64, encoder_depth=4, num_classes=2, dropout=0.5):
        super().__init__()
        
        self.encoder = nn.ModuleList()
        for i in range(0,encoder_depth):
            self.encoder.append(EncoderBlock(channels_out*2**i))
            
        self.bridge = nn.Sequential(
            nn.LazyConv2d(channels_out*2**encoder_depth, kernel_size=3), 
            nn.ReLU(),
            nn.LazyConv2d(channels_out*2**encoder_depth, kernel_size=3), 
            nn.ReLU(),
            nn.Dropout2d(dropout),
            )
        
        self.decoder = nn.ModuleList()
        for i in range(encoder_depth-1,-1,-1):
            self.decoder.append(DecoderBlock(channels_out*2**i))
            
        self.final = nn.LazyConv2d(num_classes, kernel_size=1)
        
    def forward(self, X):
        H = X
        skip = [] # LIFO
        
        for enc_block in self.encoder:
            H, S = enc_block(H)
            skip.append(S)
            
        H = self.bridge(H)
        
        for dec_block in self.decoder:
            H = dec_block(H, skip.pop())
            
        Y = self.final(H)
        return Y
    
    def init_weights(self, inputs):
        self.forward(inputs)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')

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
        
        self.unpoolblock = nn.Sequential(
            nn.LazyConvTranspose2d(out_channels, kernel_size=2, stride=2), 
            nn.ReLU(),
            )
        
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