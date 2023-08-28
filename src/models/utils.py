import numpy as np


class EarlyStopping:
    def __init__(self, patience: int = 5, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_metric = -np.inf if mode == "max" else np.inf

    def __call__(self, metric) -> bool:
        # New best or equal to previous score
        if (self.mode == "max" and metric >= self.best_metric) or (
            self.mode == "min" and metric <= self.best_metric
        ):
            self.best_metric = metric
            self.counter = 0

        # Worse score
        else:
            self.counter += 1
            if self.counter == self.patience:
                return True
        return False
