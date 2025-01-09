import traceback
import pytest
from click.testing import CliRunner
import numpy as np
import cv2 as cv
import hydra
from mitotem.main import cli, train

SIZE = (224, 224)


@pytest.fixture
def sphere_file_input(tmp_path):
    rng = np.random.default_rng()
    img = np.zeros(SIZE, dtype=np.float32)
    c = (112, 112)
    r = 50
    ii, jj = np.meshgrid(np.arange(SIZE[0]), np.arange(SIZE[1]), indexing="ij")
    img[(ii - c[0]) ** 2 + (jj - c[1]) ** 2 < r**2] = 1.0
    img += rng.normal(0, 0.1, img.shape)
    img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

    filename = tmp_path / "img0.tiff"
    cv.imwrite(filename, img)

    return str(filename)


@pytest.fixture
def file_input(tmp_path):
    rng = np.random.default_rng()
    img = rng.integers(0, 256, SIZE, dtype=np.uint8)
    filename = tmp_path / "img0.tiff"
    cv.imwrite(filename, img)

    return str(filename)


@pytest.fixture
def dir_input(tmp_path):
    rng = np.random.default_rng()
    for i in range(3):
        img = rng.integers(0, 256, SIZE, dtype=np.uint8)
        filename = tmp_path / f"img{i}.tiff"
        cv.imwrite(filename, img)

    return str(tmp_path)


@pytest.fixture
def out_dir(tmp_path):
    dn = tmp_path / "masks"
    dn.mkdir()

    return dn


@pytest.fixture
def runner():
    return CliRunner()


class TestPredict:
    def test_sam_prompts(self, runner, sphere_file_input, out_dir):
        args = [
            "predict",
            sphere_file_input,
            "--model",
            "sam2",
            "--maskout",
            "--js",
            '[{"point_coords": [[112,112]], "point_labels": [1]}]',
            "--output",
            str(out_dir),
        ]
        result = runner.invoke(cli, args)
        traceback.print_exception(*result.exc_info)
        assert result.exit_code == 0
        assert len(list(out_dir.iterdir())) == 1

    def test_sam_one_image(self, runner, file_input, out_dir):
        args = [
            "predict",
            file_input,
            "--model",
            "sam2",
            "--maskout",
            "--output",
            str(out_dir),
        ]
        result = runner.invoke(cli, args)
        traceback.print_exception(*result.exc_info)
        assert result.exit_code == 0
        assert len(list(out_dir.iterdir())) == 1

    def test_three_images(self, runner, dir_input, out_dir):
        args = [
            "predict",
            dir_input,
            "--model",
            "sam2",
            "--maskout",
            "--output",
            str(out_dir),
        ]
        result = runner.invoke(cli, args)
        traceback.print_exception(*result.exc_info)
        assert result.exit_code == 0
        assert len(list(out_dir.iterdir())) == 3


class TestTrain:
    def test_1_epoch(self, runner):
        args = ["train", "test_resnet50_unet"]
        result = runner.invoke(cli, args)
        traceback.print_exception(*result.exc_info)
        assert result.exit_code == 0
