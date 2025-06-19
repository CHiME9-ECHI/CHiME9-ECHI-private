import torch
from omegaconf import DictConfig


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def check_nan(tense: torch.Tensor):
    return torch.isnan(tense).any().item()


def check_cfg_item(cfg: DictConfig, name: str):
    if name in cfg:
        return cfg[name]
    else:
        return None
