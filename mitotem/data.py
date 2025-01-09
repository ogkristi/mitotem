from pathlib import Path
from typing import Any, Sequence
import torch
from torchvision import tv_tensors
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision.transforms.v2._utils import query_size
from torchvision.datasets import VisionDataset
import numpy as np
import cv2 as cv


class MitoSS(VisionDataset):
    def __init__(
        self,
        root: str = "data/processed",
        transform: T.Transform | None = None,
        indices: list | None = None,
    ):
        super().__init__()
        self.examples = [
            sorted((Path(root) / subdir).rglob("*.tif", case_sensitive=False))
            for subdir in ("images", "labels", "weights")
        ]
        self.examples = list(zip(*self.examples))

        if indices:
            self.examples = [self.examples[i] for i in indices]

        self.transform = transform

    def __getitem__(self, index: int):
        image, mask, weight = [
            cv.imread(str(path), cv.IMREAD_GRAYSCALE) for path in self.examples[index]
        ]

        # Erode all instance masks individually to force a border of two
        # background pixels between them
        instances = np.unique(mask[mask > 0])
        if len(instances) > 1:
            masks = (mask[:, :, None] == instances[None, None, :]).astype(np.uint8)
            masks = cv.erode(masks, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
            mask[:] = np.bitwise_or.reduce(masks, axis=2)

        _, mask = cv.threshold(mask, 0, 1, cv.THRESH_BINARY)

        # Training example is returned as numpy arrays when no transform was provided
        if self.transform is None:
            return image, mask, weight
        else:
            example = (
                tv_tensors.Image(image),
                tv_tensors.Mask(mask),
                tv_tensors.Mask(weight),
            )

            return self.transform(example)

    def __len__(self):
        return len(self.examples)


class Identity(T.Transform):
    def __init__(self):
        super().__init__()

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return inpt


class Zscore(T.Transform):
    def __init__(self):
        super().__init__()

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if not isinstance(inpt, tv_tensors.Image):
            return inpt

        if inpt.dtype != torch.float32:
            raise TypeError("Z-score is valid only for input type float32")

        std, mean = torch.std_mean(inpt)

        return self._call_kernel(F.normalize, inpt, mean=mean, std=std, inplace=True)


class RandomResizedCrop(T.Transform):
    def __init__(self, size: int | tuple[int], scale: tuple[float]):
        super().__init__()
        if isinstance(size, int):
            size = (size,) * 2
        self.size = size
        self.scale = scale

    def _get_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        a = torch.empty(1).uniform_(*self.scale).item()
        height = round(a * self.size[0])
        width = round(a * self.size[1])

        h, w = query_size(flat_inputs)
        top = torch.randint(0, h - height + 1, size=(1,)).item()
        left = torch.randint(0, w - width + 1, size=(1,)).item()

        return dict(top=top, left=left, height=height, width=width)

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return self._call_kernel(
            F.resized_crop,
            inpt,
            **params,
            size=self.size,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )


class ForegroundCrop(T.Transform):
    def __init__(self, minsize: int | Sequence[int]):
        super().__init__()

        self.minsize = minsize

    def _get_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        h, w = self.minsize, self.minsize
        mask = flat_inputs[1]

        y_max, x_max = mask.shape[-2:]
        y_max, x_max = y_max - h, x_max - w

        y_coord, x_coord = mask.nonzero(as_tuple=True)
        # Center crop if mask is all background
        if len(y_coord) == 0:
            x0 = (x_max - w) // 2
            y0 = (y_max - h) // 2
            h_fg = h
            w_fg = w
        else:
            x0, y0 = (torch.min(x_coord), torch.min(y_coord))
            x1, y1 = (torch.max(x_coord), torch.max(y_coord))
            h_fg = y1 - y0 + 1
            w_fg = x1 - x0 + 1

            # If foreground bbox is smaller than training patch (minsize),
            # it is enlarged symmetrically to minsize
            if h_fg < h:
                y0 = torch.clip(y0 - (h - h_fg) // 2, 0, y_max)
                h_fg = h
            if w_fg < w:
                x0 = torch.clip(x0 - (w - w_fg) // 2, 0, x_max)
                w_fg = w

        return dict(top=y0, left=x0, height=h_fg, width=w_fg)

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        return F.crop(inpt, **params)


augment = T.RandomChoice(
    [
        Identity(),
        T.RandomRotation(180),
        T.RandomAffine(degrees=0, shear=(-45, 45, -45, 45)),
        T.GaussianBlur(7, (0.1, 2.0)),
        T.RandomEqualize(1),
    ]
)


def transform(patch_size: int | tuple[int], train: bool = True):
    if isinstance(patch_size, int):
        patch_size = (patch_size,) * 2

    # Initial crop is a square with side length = hypotenuse length of final crop
    init_size = int(np.ceil(np.sqrt(patch_size[0] ** 2 + patch_size[1] ** 2)).item())

    if train:
        return T.Compose(
            [
                ForegroundCrop(init_size),
                RandomResizedCrop(init_size, (0.5, 2)),
                augment,
                T.CenterCrop(patch_size),
                T.ToDtype(
                    {tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64},
                    scale=True,
                ),
                Zscore(),
            ]
        )
    else:
        return T.Compose(
            [
                ForegroundCrop(patch_size),
                T.RandomCrop(patch_size),
                T.ToDtype(
                    {tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.int64},
                    scale=True,
                ),
                Zscore(),
            ]
        )
