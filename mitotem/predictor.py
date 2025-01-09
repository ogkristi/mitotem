from abc import ABC, abstractmethod
from typing import Any
import torch
import cv2 as cv
import numpy as np
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor


class Predictor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, src: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    @staticmethod
    def get_predictor(model: str):
        match (model):
            case "sam2":
                return Sam2Predictor()
            case _:
                raise ValueError(f"Unsupported model type: {model}")


class Sam2Predictor(Predictor):
    def __init__(self):
        cfg = "../config/sam2.1_hiera_base_plus.yaml"  # Relative to current file
        weights = "model/sam2.1_hiera_base_plus.pt"  # Relative to project root
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = build_sam2(cfg, weights, device)

    def predict(self, src: np.ndarray, **kwargs) -> np.ndarray:
        kwargs = {k: np.array(v) for k, v in kwargs.items() if v != []}

        if src.ndim == 2:
            src = cv.cvtColor(src, cv.COLOR_GRAY2RGB)

        # If no prompts were provided, use automatic mask generation
        if kwargs == {}:
            mask_generator = SAM2AutomaticMaskGenerator(self.model)
            masks = mask_generator.generate(src)
            mask = np.bitwise_or.reduce([m["segmentation"] for m in masks])
        else:
            predictor = SAM2ImagePredictor(self.model)
            predictor.set_image(src)
            mask, _, _ = predictor.predict(multimask_output=False, **kwargs)

        return mask.squeeze().astype(np.uint8)
