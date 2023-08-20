import random
from pathlib import Path
from random import choice
from typing import Any, Callable
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision import datapoints
from torchvision.datasets import VisionDataset
import cv2 as cv
import numpy as np
from config.settings import *

MITOSEM_CLASSES = ["background", "mitochondria"]


class TrivialAugmentWide(T.Transform):
    def __init__(self, bins: int = 31) -> None:
        super().__init__()
        self.bins = bins

        self.aug_space = {
            "identity": lambda x, m: x,
            "equalize": lambda x, m: F.equalize(x),
            "autocontrast": lambda x, m: F.autocontrast(x),
            "solarize": F.solarize,
            "posterize": F.posterize,
            "brightness": F.adjust_brightness,
            "contrast": F.adjust_contrast,
            "sharpness": F.adjust_sharpness,
            "rotate": F.rotate,
            "shear_x": lambda x, m: F.affine(
                x, 0.0, [0, 0], 1.0, [0.0, m], F.InterpolationMode.BILINEAR
            ),
            "shear_y": lambda x, m: F.affine(
                x, 0.0, [0, 0], 1.0, [m, 0.0], F.InterpolationMode.BILINEAR
            ),
            "translate_x": lambda x, m: F.affine(
                x, 0.0, [m, 0], 1.0, [0.0, 0.0], F.InterpolationMode.BILINEAR
            ),
            "translate_y": lambda x, m: F.affine(
                x, 0.0, [0, m], 1.0, [0.0, 0.0], F.InterpolationMode.BILINEAR
            ),
            "elastic": lambda x, m: T.ElasticTransform(m, 30.0)(x),
        }

        self.ranges = {
            "identity": (None, None),
            "equalize": (None, None),
            "autocontrast": (None, None),
            "solarize": (0, 255),
            "posterize": (2, 8),
            "brightness": (0.01, 2.0),
            "contrast": (0.01, 2.0),
            "sharpness": (0.01, 2.0),
            "rotate": (0.0, 135.0),
            "shear_x": (0.0, 0.99 * 45),
            "shear_y": (0.0, 0.99 * 45),
            "translate_x": (0, 32),
            "translate_y": (0, 32),
            "elastic": (0.0, 1500.0),
        }

    def _get_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        name = random.choice(list(self.aug_space))
        low, up = self.ranges[name]

        magnitude = None
        if low != None:
            magnitude = choice(np.linspace(low, up, dtype=type(low)))

        if name in ("rotate", "shear_x", "shear_y", "translate_x", "translate_y"):
            if random.random() > 0.5:
                magnitude *= -1

        return dict(transform=name, magnitude=magnitude)

    def _transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        transform = self.aug_space[params["transform"]]

        return transform(inpt, params["magnitude"])


class MitoSemsegDataset(VisionDataset):
    def __init__(
        self, root: str, transforms: Callable | None = None, weights: bool = False
    ):
        imgdir = Path(root) / "images"
        maskdir = Path(root) / "labels"
        self.images = sorted(imgdir.rglob("*.tif"))
        self.masks = sorted(maskdir.rglob("*.tif"))
        self.weights = None

        if weights:
            weightdir = Path(root) / "weights"
            self.weights = sorted(weightdir.rglob("*.tif"))

        self.transforms = transforms

    def __getitem__(
        self, index: int
    ) -> tuple[np.ndarray, ...] | tuple[torch.Tensor, ...]:
        image = cv.imread(str(self.images[index]), cv.IMREAD_GRAYSCALE)
        mask = cv.imread(str(self.masks[index]), cv.IMREAD_GRAYSCALE)

        if self.weights:
            weight = cv.imread(str(self.weights[index]), cv.IMREAD_GRAYSCALE)

            labels = np.unique(mask)
            if len(labels) > 2:
                labels = labels[1:]
                # Erode all instance masks individually to force a border of two
                # background pixels between them
                masks = (mask[:, :, None] == labels[None, None, :]).astype(np.uint8)
                masks = cv.erode(
                    masks, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
                )
                mask = np.bitwise_or.reduce(masks, axis=2)

        _, mask = cv.threshold(mask, 0, 1, cv.THRESH_BINARY)

        if self.transforms:
            example = (datapoints.Image(image), datapoints.Mask(mask, dtype=torch.long))

            if self.weights:
                example += (datapoints.Mask(weight, dtype=torch.float32),)

            example = self.transforms(example)
        else:
            example = (image, mask)

            if self.weights:
                example += (weight,)

        return example

    def __len__(self) -> int:
        return len(self.images)


def load_data_mitosemseg(data_dir: str, crop_size: int, split: float = 0.85):
    tf_train = T.Compose(
        [
            T.RandomCrop(crop_size),
            TrivialAugmentWide(),
            T.ConvertDtype(torch.float32),
            T.Normalize(mean=[0.6552], std=[0.1531]),
        ]
    )
    tf_val = T.Compose(
        [
            T.RandomCrop(crop_size),
            T.ConvertDtype(torch.float32),
            T.Normalize(mean=[0.6552], std=[0.1531]),
        ]
    )

    train = MitoSemsegDataset(root=data_dir, transforms=tf_train)
    val = MitoSemsegDataset(root=data_dir, transforms=tf_val)

    end = int(split * len(train))
    indices = torch.randperm(len(train)).tolist()
    train = torch.utils.data.Subset(train, indices[:end])
    val = torch.utils.data.Subset(val, indices[end:])

    return train, val


def get_mean_and_std() -> tuple[float, float]:
    # Dataset for getting images as float tensors scaled to [0,1]
    dataset = MitoSemsegDataset(
        root=TRAIN_ROOT, transforms=T.ConvertDtype(torch.float32)
    )

    pixels_total = 0
    sum_ = torch.zeros(1)
    sum_2 = torch.zeros(1)
    for image, _ in dataset:
        _, h, w = image.shape
        pixels_total += h * w
        sum_ += torch.sum(image)
        sum_2 += torch.sum(image**2)

    e_x = sum_ / pixels_total
    e_x2 = sum_2 / pixels_total

    return e_x.item(), torch.sqrt(e_x2 - e_x**2).item()


def get_weightmap(mask: np.ndarray) -> np.ndarray:
    w0 = 10
    sigma = 10

    # Weight is calculated only for a small border around foreground objects
    yv, xv = np.nonzero(cv.dilate(mask, np.ones((13, 13), dtype=np.uint8)) - mask)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # (negative) sum of distances to two nearest contours
    fun = lambda x, y: sum(
        sorted(
            [cv.pointPolygonTest(ctr, (float(x), float(y)), True) for ctr in contours]
        )[-2:]
    )
    sum_d1d2 = np.vectorize(fun)

    wmap = np.zeros_like(mask, dtype=np.float32)
    if len(contours) > 1:
        wmap[(yv, xv)] = w0 * np.exp(-0.5 * (sum_d1d2(xv, yv) / sigma) ** 2)

    return wmap.astype(np.uint8)


def export_weightmaps():
    maskdir = Path(TRAIN_ROOT) / "labels" / "martinthesis"
    weightdir = Path(TRAIN_ROOT) / "weights" / "martinthesis"
    weightdir.mkdir(parents=True, exist_ok=True)

    maskpaths = sorted(maskdir.rglob("*.tif"))
    for p in maskpaths:
        target = weightdir / p.name.replace("label", "weight")
        mask = cv.imread(str(p), cv.IMREAD_GRAYSCALE)

        labels = np.unique(mask)
        if len(labels) > 2:
            labels = labels[1:]
            # Erode all instance masks individually to force a border of two
            # background pixels between them
            masks = (mask[:, :, None] == labels[None, None, :]).astype(np.uint8)
            masks = cv.erode(masks, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
            mask = np.bitwise_or.reduce(masks, axis=2)

        weightmap = get_weightmap(mask)

        cv.imwrite(str(target), weightmap)
