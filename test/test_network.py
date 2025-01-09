import torch
from mitotem.network import UNet


def test_standard_unet_shape():
    model = UNet()
    x = torch.randn(1, 1, 572, 572)
    y = model(x)

    assert y.shape == torch.Size([1, 2, 388, 388])
