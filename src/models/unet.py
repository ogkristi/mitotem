import torch
from torch import nn
from torch.nn.functional import fold, unfold
from torchvision.transforms.v2.functional import center_crop, pad


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


def predict(
    src: torch.Tensor, model: nn.Module, input_size: int, output_size: int
) -> torch.Tensor:
    if src.ndim == 2:
        src = src.unsqueeze(0)
    if src.ndim == 3:
        src = src.unsqueeze(0)

    k, s = input_size, output_size
    h, w = src.shape[-2:]
    p = (input_size - output_size) // 2  # padding
    # extra one-sided paddings to make sure src size is integer multiple of stride s
    extra_w = (1 + w // output_size) * output_size - w
    extra_h = (1 + h // output_size) * output_size - h

    src = pad(
        src, [p + extra_w, p, p, p + extra_h], padding_mode="reflect"
    )  # left, top, right,  bottom

    use_fold = False  # Two alternative ways are implemented, for loop based was faster
    if use_fold:
        # make a minibatch of patches
        patches = (
            unfold(src, kernel_size=k, stride=s).permute(2, 0, 1).reshape(-1, 1, k, k)
        )
        # do prediction on minibatch
        patches = model(patches).argmax(dim=1, keepdim=True).to(torch.float32)
        patches = patches.reshape(-1, 1, s * s).permute(1, 2, 0)
        # fold patches back to full image
        dst = fold(
            patches, output_size=(h + extra_h, w + extra_w), kernel_size=s, stride=s
        )
    else:
        L_w = 1 + ((src.shape[3] - k) // s)  # Number of horizontal overlap patches
        L_h = 1 + ((src.shape[2] - k) // s)  # Number of vertical overlap patches
        L = L_h * L_w  # Total number of patches

        # Arrange patches as minibatch
        patches_in = torch.empty((L, 1, k, k), dtype=torch.float32, device=src.device)
        for i in range(L_h):
            for j in range(L_w):
                patches_in[i * L_w + j, 0, :, :] = src[
                    0, 0, i * s : i * s + k, j * s : j * s + k
                ]

        # Do prediction one row of overlap patches at a time
        patches_out = torch.empty((L, 1, s, s), dtype=torch.int64, device=src.device)
        for i in range(L_h):
            batch = patches_in[i * L_w : (i + 1) * L_w, 0, :, :]
            patches_out[i * L_w : (i + 1) * L_w, 0, :, :] = model(batch).argmax(
                dim=1, keepdim=True
            )

        # Gather patches back to a single image
        dst = torch.empty(
            (1, 1, h + extra_h, w + extra_w), dtype=torch.int64, device=src.device
        )
        for i in range(L_h):
            for j in range(L_w):
                dst[0, 0, i * s : (1 + i) * s, j * s : (1 + j) * s] = patches_out[
                    i * L_w + j, 0, :, :
                ]

    return dst[:, :, :-extra_h, extra_w:].squeeze()
