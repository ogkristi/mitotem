from math import inf
from pathlib import Path
from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from mitotem.util import Logger


class MaxEpochs:
    def __init__(self, max_epochs: int) -> None:
        self.counter = max_epochs

    def __call__(self, metric: float):
        self.counter -= 1
        return True if self.counter == 0 else False


class EarlyStop:
    def __init__(self, patience: int, mode: str = "min") -> None:
        self.patience = patience
        self.mode = mode
        self.counter = patience
        self.best_metric = -inf if mode == "max" else inf

    def __call__(self, metric: float):
        if (self.mode == "max" and metric > self.best_metric) or (
            self.mode == "min" and metric < self.best_metric
        ):
            self.best_metric = metric
            self.counter = self.patience
        else:
            self.counter -= 1
            if self.counter == 0:
                return True

        return False


class Trainer:
    def __init__(
        self,
        rundir: str,
        logger: Logger,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: _Loss,
        stop_criterion: Callable,
        amp: bool = True,
    ) -> None:
        self.dir = Path(rundir)
        self.dir.mkdir(exist_ok=True)
        self.logger = logger
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.stop_criterion = stop_criterion
        self.amp = amp

        self.epoch = 1
        self.top_loss = inf
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_fn.to(self.device)
        self.scaler = torch.GradScaler(self.device.type, enabled=self.amp)
        self.load_checkpoint()

    def load_checkpoint(self):
        checkpoint_path = self.dir / "checkpoint.pt"
        if checkpoint_path.exists():
            self.logger.info("Loading checkpoint")

            checkpoint = torch.load(checkpoint_path)
            self.epoch = checkpoint["epoch"]
            self.top_loss = checkpoint["loss"]
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scaler.load_state_dict(checkpoint["scaler"])

    def save_checkpoint(self):
        self.logger.info("Saving checkpoint")
        checkpoint = {
            "epoch": self.epoch,
            "loss": self.top_loss,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        torch.save(checkpoint, self.dir / "checkpoint.pt")

    def batch_step(
        self, train: bool, input: Tensor, target: Tensor, weight: Tensor | None = None
    ) -> Tensor:
        with torch.set_grad_enabled(train):
            self.model.train(train)
            weight = weight if isinstance(weight, Tensor) else torch.tensor(0.0)
            input, target, weight = (
                input.to(self.device),
                target.to(self.device),
                weight.to(self.device),
            )

            with torch.autocast(self.device.type, enabled=self.amp):
                prediction = self.model(input)
                loss = ((1 + weight) * self.loss_fn(prediction, target)).mean()

            if train:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

        return loss.detach()

    def epoch_step(self, loader: DataLoader, train: bool) -> float:
        loss_acc = torch.tensor(0.0)
        for batch in loader:
            loss_acc += self.batch_step(train, *batch)
        loss_mean = loss_acc.item() / len(loader)

        self.logger.scalar(self.epoch, f"Loss/{'train' if train else 'val'}", loss_mean)

        return loss_mean

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        while True:
            _ = self.epoch_step(train_loader, train=True)
            loss_val = self.epoch_step(val_loader, train=False)

            if loss_val < self.top_loss:
                self.top_loss = loss_val
                self.save_checkpoint()

            if self.stop_criterion(loss_val):
                break

            self.epoch += 1
