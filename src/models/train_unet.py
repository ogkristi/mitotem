from dotenv import load_dotenv, find_dotenv
import os, sys

load_dotenv(find_dotenv())
sys.path.append(os.getenv("PYTHONPATH"))
from config.settings import *

from pathlib import Path
import click
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as TF
from torchmetrics.classification import BinaryJaccardIndex
from torch.utils.tensorboard import SummaryWriter
from numpy import mean
from src.models.unet import UNet, find_next_valid_size, predict
from src.models.utils import EarlyStopping
from src.data.loaders import MitoSemsegDataset, TrivialAugmentWide


def load_data(data_dir: str, crop_size: int, split: float = 0.85):
    tf_train = T.Compose(
        [
            T.RandomCrop(crop_size),
            TrivialAugmentWide(),
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
    net(torch.ones((1, 1, input_size, input_size), device=device))  # dry run

    optims = {"adam": optim.Adam, "nadam": optim.NAdam, "rmsprop": optim.RMSprop}
    optimizer = optims[config["optimizer"]](
        net.parameters(), config["lr"], weight_decay=config["weight_decay"]
    )

    criterion = nn.CrossEntropyLoss(reduction="none").to(device)

    start_epoch = 1
    checkpoint_path = Path(run_dir) / "checkpoint.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint["epoch"]
        net.load_state_dict(checkpoint["net_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print("Started loading data")
    # Validation set is 16 images
    trainset, valset = load_data(data_dir, input_size, split=0.9375)
    print("Finished loading data")

    train_iter = DataLoader(
        trainset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=15,
        pin_memory=True,
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
            targets = TF.center_crop(targets, output_size)
            weights = TF.center_crop(weights, output_size)
            inputs, targets, weights = (
                inputs.to(device),
                targets.to(device),
                weights.to(device),
            )

            optimizer.zero_grad(set_to_none=True)

            outputs = net(inputs)
            loss = ((1 + weights) * criterion(outputs, targets)).mean()
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, minibatch {i}, loss {loss.item()}")
            # prof.step()

        # Evaluation of mean validation accuracy
        print("Evaluating accuracy")
        accuracies = []
        net.eval()
        for inputs, targets in val_iter:
            with torch.no_grad():
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = predict(inputs, net, input_size, output_size)
                accuracies.append(metric(outputs, targets.squeeze()).item())
        accuracy = mean(accuracies)

        # Checkpoint best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                {
                    "epoch": epoch,
                    "net_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "accuracy": accuracy,
                },
                checkpoint_path,
            )

        # Reporting
        print(
            f"Finished epoch {epoch}: training loss {mean(losses)}, validation accuracy {accuracy}"
        )
        writer.add_scalar("Loss/train", mean(losses), epoch)
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
        "batch_size": 8,
        "optimizer": "adam",
        "lr": 1e-4,
        "dropout": 0.2,
        "weight_decay": 1e-4,
        "channels_out": 56,
        "kernel_size": 5,
        "encoder_depth": 4,
    }
    train_unet(config, data_dir, run_dir)


if __name__ == "__main__":
    main()
