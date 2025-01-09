from collections import OrderedDict
from math import ceil, inf
import torch
from torch import nn
from torch.nn.functional import fold, unfold
from torchvision.transforms.v2.functional import center_crop, pad
from torchvision.models import resnet50
from mitotem.layer import ConvBlock, DoubleConv, Bottleneck


class ResNet50UNet(nn.Module):
    def __init__(self, num_classes: int = 2, resnet50_weights: str | None = None):
        super().__init__()
        resnet = resnet50()
        resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        resnet = nn.Sequential(OrderedDict(list(resnet.named_children())[:-2]))

        if resnet50_weights is not None:
            checkpoint = torch.load(resnet50_weights)
            resnet.load_state_dict(checkpoint["state_dict"])

        self.head = nn.Sequential(*list(resnet.children())[:4])
        self.encoder = nn.ModuleList(list(resnet.children())[4:])

        stages = [
            self.encoder[i][-1].conv3.out_channels
            for i in range(len(self.encoder) - 2, -1, -1)
        ]

        self.decoder = nn.ModuleList()
        for channels in stages:
            double_bottleneck = nn.ModuleList(
                [
                    Bottleneck(channels, 3, 0.5, "same"),
                    Bottleneck(channels, 3, 1, "same"),
                ]
            )
            self.decoder.append(double_bottleneck)

        self.tail = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConvTranspose2d(64, 3, stride=2, padding=1, output_padding=1),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConvTranspose2d(
                num_classes, 7, stride=2, padding=3, output_padding=1
            ),
        )

    def forward(self, X):
        H = self.head(X)

        skip = []
        for enc_block in self.encoder:
            H = enc_block(H)
            skip.append(H)
        skip.pop()

        for dec_block in self.decoder:
            H = dec_block[0](H)
            H = dec_block[1](H + skip.pop())

        Y = self.tail(H)
        return Y


class UNet(nn.Module):
    def __init__(
        self,
        Block: type[ConvBlock] = DoubleConv,
        channels_out: int = 64,
        kernel_size: int = 3,
        padding: str = "valid",
        encoder_depth: int = 4,
        num_classes: int = 2,
        dropout: float = 0.5,
        cap: int = inf,
    ):
        super().__init__()

        self.encoder = nn.ModuleList()
        for i in range(0, encoder_depth):
            self.encoder.append(
                EncoderBlock(
                    Block(min(channels_out * 2**i, cap), kernel_size, 1, padding)
                )
            )

        self.bridge = nn.Sequential(
            Block(
                min(channels_out * 2**encoder_depth, cap),
                kernel_size,
                stride=1,
                padding=padding,
            ),
            nn.Dropout2d(dropout),
        )

        self.decoder = nn.ModuleList()
        for i in range(encoder_depth - 1, -1, -1):
            self.decoder.append(
                DecoderBlock(
                    Block(min(channels_out * 2**i, cap), kernel_size, 1, padding)
                )
            )

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
    def __init__(self, convblock: ConvBlock):
        super().__init__()
        self.convblock = convblock
        self.poolblock = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, X):
        skip = self.convblock(X)
        Y = self.poolblock(skip)
        return Y, skip


class DecoderBlock(nn.Module):
    def __init__(self, convblock: ConvBlock, skip_mode: str = "cat"):
        super().__init__()
        if not skip_mode in ("cat", "sum"):
            raise ValueError('skip_mode must be one of ("cat","sum")')
        self.skip_mode = skip_mode

        self.unpoolblock = nn.Sequential(
            nn.LazyConvTranspose2d(convblock.out_channels, kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.convblock = convblock

    def forward(self, X, skip):
        Y = self.unpoolblock(X)
        if self.skip_mode == "sum":
            Y += center_crop(skip, Y.shape[-2:])
        else:
            Y = torch.cat((Y, center_crop(skip, Y.shape[-2:])), dim=1)
        Y = self.convblock(Y)
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
    extra_w = ceil(w / output_size) * output_size - w
    extra_h = ceil(h / output_size) * output_size - h

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
        L8 = ceil(L / 8) * 8

        # Arrange patches as minibatch
        patches_in = torch.empty((L8, 1, k, k), dtype=torch.float32, device=src.device)
        for i in range(L_h):
            for j in range(L_w):
                patches_in[i * L_w + j, 0, :, :] = src[
                    0, 0, i * s : i * s + k, j * s : j * s + k
                ]

        # Do prediction 8 overlap patches at a time
        patches_out = torch.empty((L8, 1, s, s), dtype=torch.int64, device=src.device)
        for i in range(0, L, 8):
            batch = patches_in[i : i + 8, :, :, :]
            patches_out[i : i + 8, :, :, :] = model(batch).argmax(dim=1, keepdim=True)

        # Gather patches back to a single image
        dst = torch.empty(
            (1, 1, h + extra_h, w + extra_w), dtype=torch.int64, device=src.device
        )
        for i in range(L_h):
            for j in range(L_w):
                dst[0, 0, i * s : (1 + i) * s, j * s : (1 + j) * s] = patches_out[
                    i * L_w + j, 0, :, :
                ]

    return dst[:, :, :-extra_h, extra_w:].squeeze(0)
