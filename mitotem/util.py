import sys
import logging
from pathlib import PosixPath
from abc import ABC, abstractmethod
from torch.utils.tensorboard import SummaryWriter


class Logger(ABC):
    @abstractmethod
    def info(self, msg: str):
        pass

    @abstractmethod
    def scalar(self, global_step: int, tag: str, scalar: float):
        pass


class ConsoleLogger(Logger):
    def __init__(self) -> None:
        formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
        consolehandler = logging.StreamHandler(sys.stdout)
        consolehandler.setFormatter(formatter)

        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(consolehandler)

    def info(self, msg: str):
        self.logger.info(msg)

    def scalar(self, global_step: int, tag: str, scalar: float):
        msg = f"Epoch {global_step}: {tag} {scalar:.3f}"
        self.logger.info(msg)


class TensorboardLogger(Logger):
    def __init__(self, dir: PosixPath) -> None:
        self.logger = SummaryWriter(dir)

    def info(self, msg: str):
        pass

    def scalar(self, global_step: int, tag: str, scalar: float):
        self.logger.add_scalar(tag, scalar, global_step)
        self.logger.flush()
