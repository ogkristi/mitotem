import torch
from torch import nn
from torchvision.transforms.v2.functional import center_crop


class UNet(nn.Module):
    def __init__(
        self,
        channels_out=64,
        kernel_size=3,
        encoder_depth=4,
        num_classes=2,
        dropout=0.5,
    ):
        super().__init__()

        self.encoder = nn.ModuleList()
        for i in range(0, encoder_depth):
            self.encoder.append(EncoderBlock(channels_out * 2**i, kernel_size))

        self.bridge = nn.Sequential(
            nn.LazyConv2d(channels_out * 2**encoder_depth, kernel_size),
            nn.ReLU(),
            nn.LazyConv2d(channels_out * 2**encoder_depth, kernel_size),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )

        self.decoder = nn.ModuleList()
        for i in range(encoder_depth - 1, -1, -1):
            self.decoder.append(DecoderBlock(channels_out * 2**i, kernel_size))

        self.final = nn.LazyConv2d(num_classes, kernel_size=1)

    def forward(self, X):
        H = X
        skip = []  # LIFO

        for enc_block in self.encoder:
            H, S = enc_block(H)
            skip.append(S)

        H = self.bridge(H)

        for dec_block in self.decoder:
            H = dec_block(H, skip.pop())

        Y = self.final(H)
        return Y


class EncoderBlock(nn.Module):
    def __init__(self, out_channels, kernel_size=3, num_convs=2):
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
    def __init__(self, out_channels, kernel_size=3, num_convs=2):
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
        H = self.unpoolblock(X)
        H = torch.cat((H, center_crop(skip, H.shape[-2:])), dim=1)
        Y = self.convblock(H)
        return Y


def find_next_valid_size(size: int, kernel_size: int, depth: int) -> tuple:
    """Searches a valid input/output size for UNet using unpadded convolutions

    Args:
        size (int): Size to begin search from going upwards
        kernel_size (int): UNet convolution kernel size
        depth (int): UNet number of downsamplings

    Returns:
        tuple: Input size, Output size
    """
    i = torch.arange(1, depth + 1)
    shrinkage = sum((kernel_size - 1) * 2**i).item()
    total_shrinkage = 2 * shrinkage + (kernel_size - 1) * 2 ** (depth + 1)
    while True:
        size_after = size - shrinkage

        if size_after % 2**depth == 0:
            return size, size - total_shrinkage
        size += 1
