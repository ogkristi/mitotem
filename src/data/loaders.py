import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import cv2 as cv
from config.settings import *

MITOSEM_CLASSES = ['background','mitochondria']

class MitoSemsegDataset(Dataset):
    def __init__(self, transform: transforms.Compose = None) -> None:
        imgdir = TRAIN_ROOT / 'images'
        labeldir = TRAIN_ROOT / 'labels'
        self.images = sorted(imgdir.rglob('*.tif'))
        self.labels = sorted(labeldir.rglob('*.tif'))
        self.transform = transforms.ToTensor() if transform is None else transform
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = cv.imread(str(self.images[index]), cv.IMREAD_GRAYSCALE)
        label = cv.imread(str(self.labels[index]), cv.IMREAD_GRAYSCALE)
        label[label > 0] = 1 # Don't care about instances, set all mitos as class 1

        return (self.transform(image), self.transform(label))

    def __len__(self) -> int:
        return len(self.images)
    
def load_data_mitosemseg(batch_size: int, num_workers: int, split: float = .85):
    train, val = random_split(MitoSemsegDataset(), [split, 1-split])
    
    train_iter = DataLoader(train, batch_size, shuffle=True, num_workers=num_workers)
    val_iter = DataLoader(val, batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, val_iter