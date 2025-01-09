import pytest
import logging
import torch
import torch.nn as nn
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torch.optim import RMSprop
import torchvision.transforms.v2 as T
from mitotem.train import MaxEpochs, EarlyStop, Trainer
from mitotem.layer import Mobile
from mitotem.util import ConsoleLogger


class TestEarlyStop:
    @pytest.fixture
    def x(self):
        return torch.arange(0, 10, step=0.1)

    @pytest.mark.parametrize("f,stop", [(lambda k: -k, 100), (lambda k: k, 11)])
    def test_monotonous_loss(self, x, f, stop):
        limit = 10
        stop_criterion = EarlyStop(limit)

        for i, loss in enumerate(f(x), start=1):
            if stop_criterion(loss.item()):
                break

        assert i == stop


class TestTrainer:
    @pytest.fixture
    def data(self):
        tx = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
        train_set = FashionMNIST("test/data", train=True, transform=tx, download=True)
        val_set = FashionMNIST("test/data", train=False, transform=tx)

        train_loader = DataLoader(
            train_set, batch_size=512, shuffle=True, drop_last=True, num_workers=1
        )
        val_loader = DataLoader(val_set, batch_size=512, shuffle=False, num_workers=1)

        return train_loader, val_loader

    @pytest.fixture
    def model(self):
        return nn.Sequential(
            Mobile(out_channels=16, stride=2),
            Mobile(out_channels=16, stride=1),
            Mobile(out_channels=32, stride=2),
            Mobile(out_channels=32, stride=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(10),
        ).float()

    def test_train_1_epoch(self, caplog, tmp_path, model, data):
        caplog.set_level(logging.INFO)
        logger = ConsoleLogger()
        model(torch.randn(1, 1, 28, 28, dtype=torch.float32))  # dry run
        optimizer = RMSprop(
            model.parameters(), lr=0.045, weight_decay=4e-05, momentum=0.9
        )
        loss_fn = nn.CrossEntropyLoss()
        trainer = Trainer(
            str(tmp_path), logger, model, optimizer, loss_fn, MaxEpochs(1), False
        )
        trainer.fit(*data)

        assert (tmp_path / "checkpoint.pt").exists()
