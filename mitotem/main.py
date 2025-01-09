import json
from pathlib import Path
import click
import cv2 as cv
import numpy as np
from mitotem.predictor import Predictor
import torch
from torch.utils.data import DataLoader
import hydra
import segmentation_models_pytorch as smp
from mitotem.data import MitoSS, transform
from mitotem.train import Trainer, EarlyStop, MaxEpochs
from mitotem.util import TensorboardLogger


def read_images(paths: list[Path]) -> list[np.ndarray]:
    image_paths = []
    for p in paths:
        if p.is_dir():
            image_paths.extend(p.rglob("*.tif", case_sensitive=False))
            image_paths.extend(p.rglob("*.tiff", case_sensitive=False))
        elif p.suffix.lower() in [".tif", ".tiff"]:
            image_paths.append(p)

    images = [cv.imread(str(p), cv.IMREAD_GRAYSCALE) for p in image_paths]
    names = [p.name for p in image_paths]

    return images, names


@click.group()
def cli():
    pass


@cli.command()
@click.argument(
    "paths",
    type=click.Path(exists=True, path_type=Path),
    nargs=-1,
)
@click.option(
    "--model",
    "-m",
    type=click.Choice(["sam2"], case_sensitive=False),
    help="Model to use for prediction",
)
@click.option("--maskout", is_flag=True, help="Write masks to files")
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path.cwd(),
    help="Destination path for output files",
)
@click.option(
    "--js", "-j", type=str, help="Optional keyword arguments to model as JSON string"
)
def predict(paths: list[Path], model: str, maskout: bool, output: Path, js: str):
    images, names = read_images(paths)

    prompts = [{}] * len(images)
    if js:
        prompts = json.loads(js)

    predictor = Predictor.get_predictor(model)

    masks = [predictor.predict(i, **p) for i, p in zip(images, prompts)]

    if maskout:
        for mask, name in zip(masks, names):
            cv.imwrite(output / name, mask)


@cli.command()
@click.argument("config", type=str, nargs=1)
def train(config: str):
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(version_base=None, config_path="../config")
    cfg = hydra.compose(config_name=config)

    train_root = cfg["data"]["train_root"]
    split = cfg["data"]["split"]
    patch_size = cfg["data"]["patch_size"]
    batch_size = cfg["data"]["batch_size"]
    num_workers = cfg["data"]["num_workers"]

    num_examples = len(MitoSS(train_root))
    bound = int(split * num_examples)
    gene = torch.Generator().manual_seed(33)
    indices = torch.randperm(num_examples, generator=gene).tolist()

    train_set = MitoSS(train_root, transform(patch_size, train=True), indices[:bound])
    val_set = MitoSS(train_root, transform(patch_size, train=False), indices[bound:])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    encoder_weights = cfg["model"]["encoder_weights"]
    num_classes = cfg["model"]["num_classes"]

    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=1,
        classes=num_classes,
    )
    if encoder_weights:
        state_dict = torch.load(encoder_weights)["state_dict"]
        state_dict = {f"encoder.{k}": v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    run_dir = cfg["train"]["run_dir"]
    max_epochs = cfg["train"]["max_epochs"]
    base_lr = cfg["train"]["base_lr"]
    weight_decay = cfg["train"]["weight_decay"]
    use_amp = cfg["train"]["use_amp"]

    trainer = Trainer(
        run_dir,
        TensorboardLogger(run_dir),
        model,
        torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay),
        torch.nn.CrossEntropyLoss(reduction="none"),
        MaxEpochs(max_epochs) if max_epochs != 0 else EarlyStop(5, "min"),
        use_amp,
    )

    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    cli()
