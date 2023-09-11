from dotenv import load_dotenv, find_dotenv
import os, sys

load_dotenv(find_dotenv())
sys.path.append(os.getenv("PYTHONPATH"))
from config.settings import *

from pathlib import Path
import time
import click
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
from torchmetrics.classification import BinaryJaccardIndex
from torch.utils.tensorboard import SummaryWriter
from numpy import mean
from src.models.unet import UNet, find_next_valid_size, predict
from src.models.utils import EarlyStopping
from src.data.loaders import (
    MitoSemsegDataset,
    TrivialAugmentWide,
    CenterCropMask,
    ForegroundCrop,
)


def load_data(data_dir: str, input_size: int, output_size: int, split: float = 0.85):
    tf_train = T.Compose(
        [
            ForegroundCrop(minsize=input_size),
            T.RandomCrop(input_size),
            TrivialAugmentWide(),
            CenterCropMask(output_size),
            T.ConvertDtype(torch.float32),
            T.Normalize(mean=[MEAN], std=[STD]),
        ]
    )
    tf_val = T.Compose(
        [
            T.ConvertDtype(torch.float32),
            T.Normalize(mean=[MEAN], std=[STD]),
        ]
    )

    bound = int(split * DATASET_SIZE)
    gene = torch.Generator().manual_seed(33)
    indices = torch.randperm(DATASET_SIZE, generator=gene).tolist()

    train = MitoSemsegDataset(
        root=data_dir, transforms=tf_train, weights=True, indices=indices[:bound]
    )
    val = MitoSemsegDataset(root=data_dir, transforms=tf_val, indices=indices[bound:])

    return train, val


def get_checkpoint(run_dir: str):
    checkpoint_path = Path(run_dir) / "checkpoint.pt"
    if checkpoint_path.exists():
        return torch.load(checkpoint_path)
    return None


def train_unet(config, data_dir, run_dir):
    Path(run_dir).mkdir(exist_ok=True)
    writer = SummaryWriter(run_dir)

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    net = UNet(
        channels_out=config["channels_out"],
        kernel_size=config["kernel_size"],
        encoder_depth=config["encoder_depth"],
        dropout=config["dropout"],
    ).to(device, torch.float32)

    input_size, output_size = find_next_valid_size(
        1000, config["kernel_size"], config["encoder_depth"]
    )
    net(torch.ones((8, 1, input_size, input_size), device=device))  # dry run

    optims = {"adam": optim.Adam, "nadam": optim.NAdam, "rmsprop": optim.RMSprop}
    optimizer = optims[config["optimizer"]](
        net.parameters(), config["lr"], weight_decay=config["weight_decay"]
    )

    scaler = torch.cuda.amp.GradScaler(enabled=config["amp"])

    start_epoch = 1
    # Load state from checkpoint if exists
    checkpoint = get_checkpoint(run_dir)
    if checkpoint:
        print("Loading checkpoint")
        start_epoch = checkpoint["epoch"]
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])

    loss_fn = nn.CrossEntropyLoss(
        weight=1 / torch.tensor(CLASS_FREQ), reduction="none"
    ).to(device)

    print("Started loading data")
    # Validation set is 16 images
    trainset, valset = load_data(data_dir, input_size, output_size, split=0.9375)
    print("Finished loading data")

    train_iter = DataLoader(
        trainset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=10,
        pin_memory=True,
        drop_last=True,
    )
    val_iter = DataLoader(
        valset,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
    )

    stop_condition = EarlyStopping(patience=10, mode="max")
    metric = BinaryJaccardIndex().to(device)
    best_accuracy = -1
    for epoch in range(start_epoch, 257):  # max epochs is 256
        timer_epoch = time.time()
        net.train()
        losses = []

        # with torch.profiler.profile(
        #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler(
        #         "/scratch/project_2008180/profiler"
        #     ),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=False,
        # ) as prof:
        # Training loop
        for i, (inputs, targets, weights) in enumerate(train_iter, 1):
            timer_batch = time.time()
            inputs, targets, weights = (
                inputs.to(device),
                targets.to(device),
                weights.to(device),
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device, torch.float16, config["amp"]):
                outputs = net(inputs)
                loss = ((1 + weights) * loss_fn(outputs, targets)).mean()
            losses.append(loss.detach())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            print(f"Epoch {epoch}, minibatch {i} ({time.time()-timer_batch:.3f}s)")
            # prof.step()

        # Evaluation of mean validation accuracy
        print("Evaluating accuracy")
        accuracies = []
        net.eval()
        for inputs, targets in val_iter:
            with torch.no_grad(), torch.autocast(device, torch.float16, config["amp"]):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = predict(inputs, net, input_size, output_size)
                accuracies.append(metric(outputs, targets))

        accuracy = torch.stack(accuracies).mean().item()
        loss = torch.stack(losses).mean().item()

        # Checkpoint best model
        if accuracy > best_accuracy:
            print("Saving checkpoint")
            best_accuracy = accuracy
            torch.save(
                {
                    "epoch": epoch,
                    "net": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "accuracy": accuracy,
                    "scaler": scaler.state_dict(),
                },
                checkpoint_path,
            )

        # Reporting
        print(
            f"Finished epoch {epoch}: training loss {loss:.3f}, "
            f"validation accuracy {accuracy:.3f} ({time.time()-timer_epoch:.3f}s)"
        )
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Accuracy/test", accuracy, epoch)

        if stop_condition(accuracy):
            break

    print("Finished training")
    writer.flush()
    writer.close()


@click.command()
@click.option("--data_dir", help="Path to training data directory")
@click.option(
    "--run_dir", default="./train_unet", help="Where to save checkpoints and logs"
)
def main(data_dir, run_dir):
    config = {
        "amp": True,  # automatic mixed precision
        "batch_size": 8,
        "optimizer": "adam",
        "lr": 1e-4,
        "dropout": 0.2,
        "weight_decay": 1e-4,
        "channels_out": 64,
        "kernel_size": 3,
        "encoder_depth": 4,
    }
    train_unet(config, data_dir, run_dir)


if __name__ == "__main__":
    main()
