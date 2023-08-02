import torch
from torch import nn
import src.models.layers as layers

class UNet(nn.Module):
    def __init__(self, channels_out=64, encoder_depth=4, num_classes=2):
        super().__init__()
        
        self.encoder = nn.ModuleList()
        for i in range(0,encoder_depth):
            self.encoder.append(layers.EncoderBlock(channels_out*2**i))
            
        self.bridge = nn.Sequential(
            nn.LazyConv2d(channels_out*2**encoder_depth, kernel_size=3), nn.ReLU(),
            nn.LazyConv2d(channels_out*2**encoder_depth, kernel_size=3), nn.ReLU(),            
        )
        
        self.decoder = nn.ModuleList()
        for i in range(encoder_depth-1,-1,-1):
            self.decoder.append(layers.DecoderBlock(channels_out*2**i))
            
        self.final = nn.LazyConv2d(num_classes, kernel_size=1)
        
    def forward(self, X):
        # H = hidden layer activations
        H = X
        skip = []
        
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