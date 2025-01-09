import pytest
import torch
from mitotem.layer import DoubleConv, Residual, Bottleneck, Mobile, PatchEmbedding


@pytest.mark.parametrize(
    "in_tensor,out_channels,stride,out_shape",
    [
        # 1st encoder stage, first block
        (torch.randn(1, 1, 224, 224), 64, 2, torch.Size([1, 64, 112, 112])),
        # 1st encoder stage, second block
        (torch.randn(1, 64, 112, 112), 64, 1, torch.Size([1, 64, 112, 112])),
        # 2nd encoder stage, first block
        (torch.randn(1, 64, 112, 112), 128, 2, torch.Size([1, 128, 56, 56])),
        # 1st decoder stage, first block
        (torch.randn(1, 128, 56, 56), 64, 0.5, torch.Size([1, 64, 112, 112])),
    ],
)
class TestConv:
    def test_doubleconv(self, in_tensor, out_channels, stride, out_shape):
        doubleconv = DoubleConv(out_channels, stride=stride)
        assert doubleconv(in_tensor).shape == out_shape

    def test_residual(self, in_tensor, out_channels, stride, out_shape):
        residual = Residual(out_channels, stride=stride)
        assert residual(in_tensor).shape == out_shape

    def test_bottleneck(self, in_tensor, out_channels, stride, out_shape):
        bottleneck = Bottleneck(out_channels, stride=stride)
        assert bottleneck(in_tensor).shape == out_shape

    def test_mobile(self, in_tensor, out_channels, stride, out_shape):
        mobile = Mobile(out_channels, stride=stride)
        assert mobile(in_tensor).shape == out_shape


class TestPatchEmbedding:
    def test_shape(self):
        img_size = (224, 224)
        patch_size = (64, 64)
        num_patches = (224 // 64) ** 2
        num_hiddens = 64**2

        patcher = PatchEmbedding(img_size, patch_size, num_hiddens)
        I = torch.rand([1, 1, 224, 224], dtype=torch.float32)
        O = patcher(I)

        assert O.shape == torch.Size([1, num_patches, num_hiddens])
