from dotenv import load_dotenv, find_dotenv
import os, sys

load_dotenv(find_dotenv())
sys.path.append(os.getenv("PYTHONPATH"))
from config.settings import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.v2.functional as TF
import click
from ray import tune, air
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
from src.models.unet import UNet, find_next_valid_size
from src.data.loaders import MitoSemsegDataset, TrivialAugmentWide, CenterCropMask


def load_data(data_dir: str, input_size: int, output_size: int, split: float = 0.85):
    tf_train = T.Compose(
        [
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


def get_checkpoint():
    checkpoint = session.get_checkpoint()
    if checkpoint:
        return checkpoint.to_dict()
    return None


def train_unet(config, data_dir):
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

    start_epoch = 1
    checkpoint = get_checkpoint()
    if checkpoint:
        print("Loading checkpoint")
        start_epoch = checkpoint["epoch"]
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # mean is calculated after pixel-wise weighing
    loss_fn = nn.CrossEntropyLoss(reduction="none").cuda(device)

    trainset, valset = load_data(data_dir, input_size, output_size, split=0.84375)

    train_iter = DataLoader(
        trainset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_iter = DataLoader(
        valset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    for epoch in range(start_epoch, 11):  # max epochs is 10
        net.train()

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
            inputs, targets, weights = (
                inputs.to(device),
                targets.to(device),
                weights.to(device),
            )

            optimizer.zero_grad(set_to_none=True)

            outputs = net(inputs)
            loss = ((1 + weights) * loss_fn(outputs, targets)).mean()
            loss.backward()
            optimizer.step()

            # prof.step()

        losses = []
        net.eval()
        for i, (inputs, targets) in enumerate(val_iter, 1):
            with torch.no_grad():
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = loss_fn(outputs, targets).mean()
                losses.append(loss)
        loss = torch.stack(losses).mean().item()

        checkpoint_data = {
            "epoch": epoch,
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        loss = running_loss / len(val_iter)
        print(f"Epoch {epoch}: loss: {loss:.3f}")
        session.report({"loss": loss}, checkpoint=checkpoint)

    print("Finished training")


@click.command()
@click.option("--data_dir", help="Path to data directory")
@click.option("--num_samples", default=50, help="Times to sample the search space")
def main(data_dir, num_samples):
    config = {
        "batch_size": tune.choice([1, 2, 4]),
        "optimizer": tune.choice(["adam", "nadam", "rmsprop"]),
        "lr": tune.loguniform(1e-5, 1e-3),
        "dropout": tune.uniform(0.0, 0.5),
        "weight_decay": tune.choice([1e-3, 1e-4, 1e-5, 0]),
        "channels_out": tune.randint(32, 65),
        "kernel_size": tune.choice([3, 5]),
        "encoder_depth": tune.randint(3, 6),
    }

    scheduler = ASHAScheduler(
        grace_period=1,
        reduction_factor=2,
    )

    trainable = tune.with_resources(
        tune.with_parameters(train_unet, data_dir=data_dir),
        resources={"gpu": 1, "cpu": 1},
    )

    run_dir = "/scratch/project_2008180/hpo_unet_results"
    if not os.path.exists(run_dir):
        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                metric="loss", mode="min", scheduler=scheduler, num_samples=num_samples
            ),
            run_config=air.RunConfig(
                local_dir=os.path.dirname(run_dir), name=os.path.basename(run_dir)
            ),
            param_space=config,
        )
    else:
        tuner = tune.Tuner.restore(run_dir, trainable)

    result = tuner.fit()

    best_result = result.get_best_result("loss", "min", "last")
    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final validation loss: {best_result.last_result['loss']}")


if __name__ == "__main__":
    main()
