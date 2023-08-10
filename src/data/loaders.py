import random
from pathlib import Path
from random import choice
from typing import Any, Optional, Callable, Tuple, Dict, List
import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision import datapoints
from torchvision.datasets import VisionDataset
import cv2 as cv
import numpy as np
from config.settings import *

MITOSEM_CLASSES = ['background','mitochondria']

class TrivialAugmentWide(T.Transform):
    def __init__(self, bins: int = 31) -> None:
        super().__init__()
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
            'shear_x': lambda x, m: F.affine(x, 0., [0,0], 1., [0.,m], F.InterpolationMode.BILINEAR),
            'shear_y': lambda x, m: F.affine(x, 0., [0,0], 1., [m,0.], F.InterpolationMode.BILINEAR),
            'translate_x': lambda x, m: F.affine(x, 0., [m,0], 1., [0.,0.], F.InterpolationMode.BILINEAR),
            'translate_y': lambda x, m: F.affine(x, 0., [0,m], 1., [0.,0.], F.InterpolationMode.BILINEAR),
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

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        name = random.choice(list(self.aug_space))
        low, up = self.ranges[name]

        magnitude = None
        if low != None:
            magnitude = choice(np.linspace(low, up, dtype=type(low)))

        if name in ('rotate','shear_x','shear_y','translate_x','translate_y'):
            if random.random() > 0.5:
                magnitude *= -1

        return dict(transform=name, magnitude=magnitude)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        transform = self.aug_space[params['transform']]

        return transform(inpt, params['magnitude'])

class MitoSemsegDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable] = None) -> None:
        imgdir = Path(root) / 'images'
        maskdir = Path(root) / 'labels'
        self.images = sorted(imgdir.rglob('*.tif'))
        self.masks = sorted(maskdir.rglob('*.tif'))
        self.transforms = transforms
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = cv.imread(str(self.images[index]), cv.IMREAD_GRAYSCALE)
        mask = cv.imread(str(self.masks[index]), cv.IMREAD_GRAYSCALE)

        # Don't care about instances, set all mitos as class 1
        _, mask = cv.threshold(mask,thresh=0,maxval=1,type=cv.THRESH_BINARY)

        if self.transforms:
            image, mask = self.transforms(datapoints.Image(image), 
                                          datapoints.Mask(mask, dtype=torch.long))

        return image, mask

    def __len__(self) -> int:
        return len(self.images)
    
def load_data_mitosemseg(data_dir: str, split: float = 0.85):
    tf_train = T.Compose([
        TrivialAugmentWide(),
        T.ConvertDtype(torch.float32),
        T.Normalize(mean=[0.6552], std=[0.1531]),
        ])
    tf_val = T.Compose([
        T.ConvertDtype(torch.float32),
        T.Normalize(mean=[0.6552], std=[0.1531]),
        ])

    train = MitoSemsegDataset(root=data_dir, transforms=tf_train)
    val = MitoSemsegDataset(root=data_dir, transforms=tf_val)
    
    end = int(split*len(train))
    indices = torch.randperm(len(train)).tolist()
    train = torch.utils.data.Subset(train, indices[:end])
    val = torch.utils.data.Subset(val, indices[end:])    

    return train, val

def get_mean_and_std() -> Tuple[float, float]:
    dataset = MitoSemsegDataset(root=TRAIN_ROOT, transforms=T.ConvertDtype())

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