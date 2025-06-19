from omegaconf import DictConfig
from torch.utils.data import DataLoader

from utils.audio_prep import AudioPrep


def get_data_loader(
    cfg: DictConfig,
    split: str,
    debug: bool,
    noisy_prep: AudioPrep,
    ref_prep: AudioPrep,
    spk_prep: AudioPrep,
):
    """
    Get the data loader for the specified dataset and split.

    Parameters
    ----------
    cfg : DictConfig
        The configuration object containing dataset and data loader settings.
    split : str
        The split of the dataset to load (e.g., 'train', 'val', 'test').
    debug : bool
        Whether to run in debug mode (e.g., load a smaller subset of the data).

    Returns
    -------
    DataLoader
        The data loader for the specified dataset and split.
    """

    if cfg.name == "CEC2":
        from dataloaders.cec import ClarityData, collate_fn

        custom_collate = True
        dataset = ClarityData(
            cfg.name,
            cfg.metadata,
            cfg.audio_dir,
            cfg.spkadapt_dir,
            cfg.signal.noisy_sr,
            cfg[split].length,
            split,
            debug,
            noisy_prep,
            ref_prep,
        )
    elif cfg.name == "chime9_echi":
        from dataloaders.echi import ECHI, collate_fn

        custom_collate = True
        dataset = ECHI(
            "ECHI",
            cfg.metadata,
            cfg.audio_dir,
            cfg.device,
            split,
            cfg.onesession,
            debug,
            noisy_prep,
            ref_prep,
            spk_prep,
        )
    else:
        raise ValueError(f"Unknown dataset name: {cfg.dataset.name}")

    loader_cfg = cfg[split].loader
    if custom_collate:
        loader = DataLoader(dataset, **loader_cfg, collate_fn=collate_fn)
    else:
        loader = DataLoader(dataset, **loader_cfg)

    save_ids = [int(i * len(dataset)) for i in [0.25, 0.5, 0.75]]
    if len(dataset) < len(save_ids):
        save_ids = save_ids[: len(dataset)]
    save_scenes = [dataset.__getitem__(i)["id"] for i in save_ids]

    return loader, save_scenes


def get_inference_loader(cfg: DictConfig, meta: dict):
    from dataloaders.echi_inference import ECHI_Inference

    audiodev_pos = int(meta[f"{cfg.audio_device}_pos"])
    target_pids = [meta[f"pos{i}"] for i in range(1, 5) if i != audiodev_pos]

    dataset = ECHI_Inference(
        "ECHI",
        cfg.data_root,
        meta["session"],
        target_pids,
        cfg.audio_device,
        cfg.split,
        cfg.sample_rate,
        cfg.window_len,
        cfg.stride,
    )
    loader = DataLoader(dataset, num_workers=0)

    return loader
