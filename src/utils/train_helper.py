from pathlib import Path
from hydra.core.hydra_config import HydraConfig
import logging
import numpy as np
import torch
import torchaudio
import wandb

from utils.file_utils import read_json, write_json, read_txt, write_txt
from utils.run_names import RUN_NAMES


class LossTracker:
    def __init__(self):
        self.loss = torch.tensor(0.0)
        self.steps = 0
        self.history = []

    def update(self, loss: torch.Tensor):
        self.loss += loss
        self.steps += 1

    def get_average(self) -> float:
        return self.loss.item() / self.steps if self.steps > 0 else 0

    def reset(self, epoch):
        self.history.append((epoch, self.get_average()))
        self.loss = torch.tensor(0.0)
        self.steps = 0


class Helper:
    def __init__(self, epochs, loss_name, debug, wandb_entity, wandb_project):
        self.debug = debug
        self.epochs = epochs
        self.loss_name = loss_name

        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project

        self.train_loss = LossTracker()
        self.val_loss = LossTracker()
        self.val_stoi = LossTracker()

        self.output_path = Path(HydraConfig.get().runtime.output_dir)
        self.json_name = self.output_path / "train_log.json"

        self.train_sampledir = self.output_path / "train_samples"
        self.dev_sampledir = self.output_path / "val_samples"
        self.train_sampledir.mkdir(parents=True, exist_ok=True)
        self.dev_sampledir.mkdir(parents=True, exist_ok=True)

        self.ckpt_dir = self.output_path / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def write_runname(self):
        pokemon = set(RUN_NAMES)
        hydra_dir = self.output_path.parent
        other_runs = hydra_dir.glob("*")

        used_names = []
        for run in other_runs:
            name_fpath = run / "name.txt"
            if name_fpath.exists():
                used_names.append(read_txt(name_fpath))

        used_names = set(used_names)
        available_names = list(pokemon.difference(used_names))

        if len(available_names) == 0:
            raise ValueError("Run out of names!")

        name = np.random.choice(available_names, 1)[0]
        write_txt(self.output_path / "name.txt", name)

    def start_training(self):
        write_json(self.json_name, [])

        run_name = self.output_path.name
        self.write_pokemon()

        if not self.debug:
            wandb.init(
                entity=self.wandb_entity,
                project=self.wandb_project,
                name=run_name,
                dir=self.output_path,
            )

        logging.info("Training")

    def epoch_report(self, epoch, do_ckpt, model: torch.nn.Module, delim=" "):

        train_log = read_json(self.json_name)
        new_log = {
            "epoch": epoch,
            "train_loss": np.nan,
            "val_loss": np.nan,
            "val_stoi": np.nan,
        }

        wlog = {}

        train_loss = self.train_loss.get_average()
        out_string = f"{delim}Epoch {epoch+1} report"
        out_string += f"{delim}Train loss: {train_loss:.2f}"
        self.train_loss.reset(epoch)

        wlog[f"train_{self.loss_name}"] = train_loss
        new_log["train_loss"] = train_loss

        if do_ckpt:
            val_loss = self.val_loss.get_average()
            val_stoi = self.val_stoi.get_average()
            out_string += f"{delim}Val loss: {val_loss:.2f}"
            out_string += f"{delim}Val stoi: {val_stoi:.2f}"
            self.val_loss.reset(epoch)
            self.val_stoi.reset(epoch)

            wlog[f"val_{self.loss_name}"] = val_loss
            wlog["val_stoi"] = val_stoi

            new_log["val_loss"] = val_loss
            new_log["val_stoi"] = val_stoi

            ckpt_path = self.get_ckpt_path(epoch)
            torch.save(model.state_dict(), ckpt_path)

        logging.info(out_string)
        train_log.append(new_log)
        write_json(self.json_name, train_log)

        if not self.debug:
            wandb.log(wlog)

    def save_sample(
        self,
        sample: torch.Tensor,
        fs: int,
        split: str,
        epoch: int,
        scene: str,
        audio_type: str,
    ):
        if split == "train":
            audio_dir = self.train_sampledir
        elif split == "dev":
            audio_dir = self.dev_sampledir

        audio_fname = f"{scene}_{audio_type}.wav"
        if audio_type == "proc":
            audio_fname = f"epoch{str(epoch).zfill(3)}_" + audio_fname
        audio_path = audio_path = str(audio_dir / audio_fname)

        if sample.ndim == 1:
            sample = sample.unsqueeze(0)
        elif sample.ndim == 3:
            sample = sample.squeeze(0)

        torchaudio.save(audio_path, sample, fs)

    def get_ckpt_path(self, epoch):
        return self.ckpt_dir / f"epoch{str(epoch).zfill(3)}.pt"

    def set_run_name(self, tracker):
        old_names = [x["name"] for x in tracker]
        options = [x for x in RUN_NAMES if x not in old_names]

        if len(options) == 0:
            logging.error("ALL NAMES USED")

        return np.random.choice(options)
