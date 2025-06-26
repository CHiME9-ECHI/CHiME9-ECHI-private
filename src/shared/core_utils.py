from omegaconf import DictConfig
from pathlib import Path
from typing import Optional
import torch

from shared.CausalMCxTFGridNet import MCxTFGridNet


def get_model(cfg: DictConfig, ckpt_path: Optional[Path] = None) -> torch.nn.Module:

    if cfg.name == "baseline":
        model = MCxTFGridNet(**cfg.params)
    else:
        raise ValueError(f"Model {cfg.name} not recognised. Add code here!")

    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)

    return model


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
