import logging
from mitotem.util import ConsoleLogger, TensorboardLogger


class TestLogger:
    def test_consolelogger(self, caplog):
        caplog.set_level(logging.INFO)
        logger = ConsoleLogger()
        logger.info("Loading data")
        logger.scalar(1, "Loss/train", 1.0)
        logger.scalar(2, "Loss/train", 0.9)

        assert len(caplog.records) == 3
        assert "Epoch 1: Loss/train 1.000" in caplog.text

    def test_tensorboardlogger(self, tmp_path):
        logger = TensorboardLogger(tmp_path)
        logger.scalar(1, "Loss/train", 1.0)
        logger.scalar(2, "Loss/train", 0.9)

        assert len(list(tmp_path.iterdir())) == 1
