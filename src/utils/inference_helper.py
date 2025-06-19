from omegaconf import OmegaConf
from pathlib import Path
import torch

from utils.file_utils import read_json, read_txt


def get_run_dir(target_name: str, log_dir: str):

    runs = Path(log_dir).glob("*/")

    for run in runs:
        name_fpath = run / "name.txt"
        if not name_fpath.exists():
            continue

        run_name = read_txt(name_fpath)
        if run_name == target_name:
            return run

    raise ValueError(f"Run name {target_name} not found")


def get_model_info(run_dir: Path, device):
    train_log = read_json(run_dir / "train_log.json")

    best_epoch = max(train_log, key=lambda x: x["val_stoi"])

    ckpt_path = run_dir / "checkpoints" / f"epoch{str(best_epoch['epoch']).zfill(3)}.pt"
    ckpt = torch.load(ckpt_path, weights_only=True, map_location=device)

    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")

    return cfg, ckpt, best_epoch["epoch"]
