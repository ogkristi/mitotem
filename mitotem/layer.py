from abc import ABC, abstractmethod
from functools import partial
import torch.nn as nn
from torchvision.transforms.v2.functional import center_crop


class ConvBlock(nn.Module, ABC):
    def __init__(
        self, out_channels: int, kernel_size: int, stride: float | int, padding: str
    ):
        super().__init__()
        self.out_channels = out_channels

        stride_space = (0.5, 1, 2)
        if stride not in stride_space:
            raise ValueError(f"Stride must be one of {stride_space}")

        padding_space = ("same", "valid")
        if padding not in padding_space:
            raise ValueError(f"Padding must be one of {padding_space}")

        self.padding = (kernel_size - 1) // 2 if padding == "same" else 0

        if stride < 1:
            self.stride = int(1 / stride)
            self.StridedConv2d = partial(
                nn.LazyConvTranspose2d, output_padding=1, bias=False
            )
        else:
            self.stride = stride
            self.StridedConv2d = partial(nn.LazyConv2d, bias=False)

    @abstractmethod
    def forward(self, x):
        pass


class DoubleConv(ConvBlock):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 3,
        stride: float | int = 1,
        padding: str = "same",
    ):
        super().__init__(out_channels, kernel_size, stride, padding)

        self.net = nn.Sequential(
            self.StridedConv2d(out_channels, kernel_size, self.stride, self.padding),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(out_channels, kernel_size, 1, self.padding, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Residual(ConvBlock):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 3,
        stride: float | int = 1,
        padding: str = "same",
    ):
        super().__init__(out_channels, kernel_size, stride, padding)

        self.residual = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            self.StridedConv2d(out_channels, kernel_size, self.stride, self.padding),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(out_channels, kernel_size, 1, self.padding),
        )

    def forward(self, x):
        y = self.residual(x)
        if self.stride == 1 and y.shape[1] == x.shape[1]:
            y += center_crop(x, y.shape[-2:])
        return y


class Bottleneck(Residual):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 3,
        stride: float | int = 1,
        padding: str = "same",
        expansion: float | int = 0.25,
    ):
        super().__init__(out_channels, kernel_size, stride, padding)
        inner_channels = int(out_channels * expansion)

        self.residual = nn.Sequential(
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(inner_channels, 1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            self.StridedConv2d(inner_channels, kernel_size, self.stride, self.padding),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(out_channels, 1),
        )


class Mobile(Residual):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 3,
        stride: float | int = 1,
        padding: str = "same",
        expansion: float | int = 4,
    ):
        super().__init__(out_channels, kernel_size, stride, padding)
        # For practical purposes out_channels/stride = in_channels
        inner_channels = int(out_channels * expansion / stride)

        self.residual = nn.Sequential(
            nn.LazyConv2d(inner_channels, 1, bias=False),
            nn.LazyBatchNorm2d(),
            nn.ReLU6(),
            self.StridedConv2d(
                inner_channels,
                kernel_size,
                self.stride,
                self.padding,
                groups=inner_channels,
            ),
            nn.LazyBatchNorm2d(),
            nn.ReLU6(),
            nn.LazyConv2d(out_channels, 1),
        )


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: tuple[int], patch_size: tuple[int], num_hiddens: int):
        super().__init__()

        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1]
        )
        self.conv = nn.LazyConv2d(
            num_hiddens, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        return self.conv(x).flatten(2).transpose(1, 2)
