import random
from pathlib import Path
from random import choice
from functools import partial
from typing import Any, Optional, Callable, Tuple
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.datasets import VisionDataset
import cv2 as cv
import numpy as np
from config.settings import *

MITOSEM_CLASSES = ['background','mitochondria']

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label

class Normalize:
    def __init__(self):
        pass
    def __call__(self, image, label):
        image = F.normalize(image, mean=[0.6552], std=[0.1531])
        return image, label
    
class ToTensor:
    def __init__(self):
        pass
    def __call__(self, image, label):
        image = torch.from_numpy(image).unsqueeze(0)
        label = torch.from_numpy(label).to(torch.long).unsqueeze(0)
        return image, label # (1, H, W)
    
class ConvertImageDtype:
    def __init__(self, dtype: torch.dtype = torch.float32):
        self.dtype = dtype

    def __call__(self, image, label):
        image = F.convert_image_dtype(image, dtype=self.dtype)
        return image, label

class TrivialAugmentWide:
    def __init__(self, bins: int = 31) -> None:
        self.bins = bins

        self.aug_space = {
            'identity': lambda x, m: x,
            'equalize': lambda x, m: F.equalize(x),
            'autocontrast': lambda x, m: F.autocontrast(x),
            'solarize': F.solarize,
            'posterize': F.posterize,
            'brightness': F.adjust_brightness,
            'contrast': F.adjust_contrast,
            'sharpness': F.adjust_sharpness,
            'rotate': F.rotate,
            'shear_x': lambda x, m, ipol: F.affine(x, 0., [0,0], 1., [0.,m], ipol),
            'shear_y': lambda x, m, ipol: F.affine(x, 0., [0,0], 1., [m,0.], ipol),
            'translate_x': lambda x, m, ipol: F.affine(x, 0., [m,0], 1., [0.,0.], ipol),
            'translate_y': lambda x, m, ipol: F.affine(x, 0., [0,m], 1., [0.,0.], ipol),
        }

        self.ranges = {
            'identity': (None,None),
            'equalize': (None,None),
            'autocontrast': (None,None),
            'solarize': (0, 255),
            'posterize': (2, 8),
            'brightness': (0.01, 2.),
            'contrast': (0.01, 2.),
            'sharpness': (0.01, 2.),
            'rotate': (0., 135.),
            'shear_x': (0., 0.99*45),
            'shear_y': (0., 0.99*45),
            'translate_x': (0, 32),
            'translate_y': (0, 32),
        }

    def __call__(self, image, label):
        name, transform = random.choice(list(self.aug_space.items()))
        low, up = self.ranges[name]

        magnitude = None
        if low != None:
            magnitude = choice(np.linspace(low, up, dtype=type(low)))

        # Only geometric transformations are applied to label image
        if name in ('rotate','shear_x','shear_y','translate_x','translate_y'):
            if random.random() > 0.5:
                magnitude *= -1
            image = transform(image, magnitude, F.InterpolationMode.BILINEAR)
            label = transform(label, magnitude, F.InterpolationMode.NEAREST)
        else:
            image = transform(image, magnitude)

        return image, label

class MitoSemsegDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable] = None) -> None:
        imgdir = Path(root) / 'images'
        labeldir = Path(root) / 'labels'
        self.images = sorted(imgdir.rglob('*.tif'))
        self.labels = sorted(labeldir.rglob('*.tif'))
        self.transforms = transforms
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = cv.imread(str(self.images[index]), cv.IMREAD_GRAYSCALE)
        label = cv.imread(str(self.labels[index]), cv.IMREAD_GRAYSCALE)

        # Don't care about instances, set all mitos as class 1
        _, label = cv.threshold(label,thresh=0,maxval=1,type=cv.THRESH_BINARY)

        if self.transforms:
            image, label = self.transforms(image, label)

        return image, label

    def __len__(self) -> int:
        return len(self.images)
    
def load_data_mitosemseg(batch_size: int, num_workers: int, split: float = 0.85):
    train, val = random_split(MitoSemsegDataset(root=TRAIN_ROOT), [split, 1-split])
    
    train_iter = DataLoader(train, batch_size, shuffle=True, num_workers=num_workers)
    val_iter = DataLoader(val, batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, val_iter

def get_mean_and_std() -> Tuple[float, float]:
    tf = Compose([ToTensor(), ConvertImageDtype(torch.float32)])
    dataset = MitoSemsegDataset(root=TRAIN_ROOT, transforms=tf)

    pixels_total = 0
    sum_ = torch.zeros(1)
    sum_2 = torch.zeros(1)
    for image, _ in dataset:
        _, h, w = image.shape
        pixels_total += h*w
        sum_ += torch.sum(image)
        sum_2 += torch.sum(image**2)

    e_x = sum_/pixels_total
    e_x2 = sum_2/pixels_total

    return e_x.item(), torch.sqrt(e_x2 - e_x**2).item()