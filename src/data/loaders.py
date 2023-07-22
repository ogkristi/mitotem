import torch
from torch.utils.data import Dataset, DataLoader
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
    
def load_data_mitosemseg(batch_size: int = 1, split: int = 85):
    train = MitoSemsegDataset()
    val = MitoSemsegDataset()

    trainsize = split*len(train)//100
    indices = torch.randperm(len(train))
    train = torch.utils.data.Subset(train, indices[:trainsize])
    val = torch.utils.data.Subset(val, indices[trainsize:])

    train_iter = DataLoader(train, batch_size, shuffle=True)
    val_iter = DataLoader(val, batch_size, shuffle=False)

    return train_iter, val_iter