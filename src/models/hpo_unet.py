from dotenv import load_dotenv, find_dotenv
import os, sys

load_dotenv(find_dotenv())
sys.path.append(os.getenv("PYTHONPATH"))
from config.settings import *

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.v2.functional as F
import numpy as np
import click
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
from src.models.unet import UNet, find_next_valid_size
from src.data.loaders import load_data_mitosemseg


def train_unet(config):
    net = UNet(
        channels_out=config["channels_out"],
        encoder_depth=config["encoder_depth"],
        dropout=config["dropout"],
    )

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    common = {k: config[k] for k in ("lr", "weight_decay")}
    common["params"] = net.parameters()
    optims = {
        "sgd": optim.SGD(**common, momentum=config["momentum"]),
        "adam": optim.Adam(**common),
        "adamw": optim.AdamW(**common),
    }

    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = optims[config["optimizer"]]

    checkpoint = session.get_checkpoint()
    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        net.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 1

    data_dir = os.getenv("LOCAL_SCRATCH") + "/dataset"
    input_size, output_size = find_next_valid_size(1000, 3, config["encoder_depth"])
    trainset, valset = load_data_mitosemseg(data_dir, input_size, split=0.85)

    train_iter = DataLoader(
        trainset, config["batch_size"], shuffle=True, num_workers=16
    )
    val_iter = DataLoader(valset, config["batch_size"], shuffle=False, num_workers=16)

    for epoch in range(start_epoch, 11):  # max epochs is 10
        running_loss = 0.0
        epoch_steps = 0
        for i, (inputs, targets, weights) in enumerate(train_iter, 1):
            targets = F.center_crop(targets, output_size)
            weights = F.center_crop(weights, output_size)
            inputs, targets, weights = (
                inputs.to(device),
                targets.to(device),
                weights.to(device),
            )

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = ((1 + weights) * criterion(outputs, targets)).mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_steps += 1
            if i % 32 == 0:
                print(f"[{epoch}, {i:>5}] loss: {running_loss/epoch_steps:.3f}")
                running_loss = 0.0

        val_loss = 0.0
        val_steps = 0
        for i, (inputs, targets) in enumerate(val_iter, 1):
            with torch.no_grad():
                targets = F.center_crop(targets, output_size)
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = net(inputs)

                loss = criterion(outputs, targets)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report({"loss": val_loss / val_steps}, checkpoint=checkpoint)

    print("Finished training")


@click.command()
@click.option("--num_samples", default=20, help="Times to sample the search space")
@click.option("--gpus_per_trial", default=2, help="Number of GPUs to use per trial")
def main(num_samples, gpus_per_trial):
    config = {
        "batch_size": tune.grid_search([1, 2, 4, 8]),
        "optimizer": tune.grid_search(["sgd", "adam", "adamw"]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "dropout": tune.uniform(0.0, 0.5),
        "momentum": tune.sample_from(
            lambda spec: np.random.uniform(0.5, 0.99)
            if spec.config.optimizer == "sgd"
            else None
        ),
        "weight_decay": tune.grid_search([1e-3, 1e-4, 1e-5, 0]),
        "channels_out": tune.randint(32, 65),
        "encoder_depth": tune.grid_search([3, 4, 5]),
    }

    scheduler = ASHAScheduler(
        grace_period=1,
        reduction_factor=2,
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_unet),
            resources={"gpu": gpus_per_trial},
        ),
        tune_config=tune.TuneConfig(
            metric="loss", mode="min", scheduler=scheduler, num_samples=num_samples
        ),
        param_space=config,
    )

    result = tuner.fit()

    best_result = result.get_best_result("loss", "min", "last")
    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation loss: {best_result.last_result['loss']}")


if __name__ == "__main__":
    main()
