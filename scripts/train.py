import torch
from tqdm import tqdm
import logging
from torch_stoi import NegSTOILoss
from torch.utils.data.dataloader import DataLoader

from dataloaders.echi import ECHI, collate_fn
from models import get_model
from losses import get_loss
from utils.misc import check_cfg_item, get_device
from utils.train_helper import Helper
from utils.signal_utils import STFTWrapper, match_length
from utils.audio_prep import AudioPrep

torch.manual_seed(666)


def save_sample(
    sample_rate: int,
    processed: torch.Tensor,
    batch_scenes: list,
    save_scenes: list,
    split: str,
    epoch: int,
    noisy: torch.Tensor,
    target: torch.Tensor,
    gromit: Helper,
):
    saves = list(set(batch_scenes) & set(save_scenes))
    if not saves:
        return None

    processed = processed.detach().cpu()
    if epoch == 0:
        noisy = noisy.detach().cpu()
        target = target.detach().cpu()
    for i, scene in enumerate(batch_scenes):
        if scene in save_scenes:
            gromit.save_sample(
                processed[i],
                sample_rate,
                split,
                epoch,
                scene,
                "proc",
            )
            if epoch == 0:
                gromit.save_sample(
                    noisy[i],
                    sample_rate,
                    split,
                    epoch,
                    scene,
                    "noisy",
                )
                gromit.save_sample(
                    target[i],
                    sample_rate,
                    split,
                    epoch,
                    scene,
                    "target",
                )


def check_lengths(
    scene: list[str],
    processed: torch.Tensor,
    target: torch.Tensor,
    split: str,
    do_stft: bool,
):
    use_val = True
    if processed.shape[-1] != target.shape[-1]:
        len_diff = abs(processed.shape[-1] - target.shape[-1])
        if not do_stft and len_diff > 1000:
            # Difference not due to stft
            logging.error(
                f"Time samples mismatch ({split}). Batch: {scene}. Proc: {processed.shape[-1]}. Targ: {target.shape[-1]}"
            )
            use_val = False
        processed, target = match_length(processed, target)
    return processed, target, use_val


def run(paths_cfg, model_cfg, train_cfg, debug, wandb_entity=None, wandb_project=None):

    device = get_device()

    # Training helper
    gromit = Helper(
        train_cfg.epochs, train_cfg.loss.name, debug, wandb_entity, wandb_project
    )

    # Model and training bits and bobs

    if model_cfg.input.type == "stft":
        do_stft = True
        stft = STFTWrapper(**model_cfg.input.stft, device=device)
    elif model_cfg.input.type != "wave":
        logging.error(f"Unrecognised model input type {model_cfg.input.type}")
    else:
        do_stft = False

    use_spkid = check_cfg_item(model_cfg.input, "use_spkid")
    if use_spkid is None:
        use_spkid = False

    noisy_prepper = AudioPrep(
        output_channels=model_cfg.input.channels,
        input_sr=48000,
        output_sr=model_cfg.input.sample_rate,
        output_rms=model_cfg.input.rms,
        device="cpu",
    )
    target_prepper = AudioPrep(
        output_channels=1,
        input_sr=16000,
        output_sr=model_cfg.input.sample_rate,
        output_rms=model_cfg.input.rms,
        device="cpu",
    )
    spk_prepper = AudioPrep(
        output_channels=1,
        input_sr=48000,
        output_sr=model_cfg.input.sample_rate,
        output_rms=model_cfg.input.rms,
        device="cpu",
    )

    trainset = ECHI(
        "train",
        "aria",
        paths_cfg.device_segment_dir,
        paths_cfg.ref_segment_dir,
        paths_cfg.rainbow_signal_dir,
        "data/chime9_echi/sessions.{dataset}.csv",
        debug,
        noisy_prepper,
        target_prepper,
        spk_prepper,
    )
    trainsaves = [trainset.__getitem__(i)["id"] for i in range(3)]
    trainset = DataLoader(trainset, 1, True, num_workers=0, collate_fn=collate_fn)

    devset = ECHI(
        "dev",
        "aria",
        paths_cfg.device_segment_dir,
        paths_cfg.ref_segment_dir,
        paths_cfg.rainbow_signal_dir,
        "data/chime9_echi/sessions.{dataset}.csv",
        debug,
        noisy_prepper,
        target_prepper,
        spk_prepper,
    )
    devsaves = [devset.__getitem__(i)["id"] for i in range(3)]
    devset = DataLoader(devset, 1, False, num_workers=0, collate_fn=collate_fn)

    model = get_model(model_cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    loss_fn = get_loss(train_cfg.loss.name, train_cfg.loss.kwargs)
    stoi_fn = NegSTOILoss(model_cfg.input.sample_rate).to(device)
    ckpt_interval = train_cfg.checkpoint_interval

    model.to(device)
    loss_fn.to(device)

    gromit.start_training()

    # Train this fine chap
    for epoch in range(train_cfg.epochs):
        model.train()

        if debug:
            loader = tqdm(trainset, desc="Training loop")
        else:
            loader = trainset

        for batch in loader:
            noisy = batch["noisy"]
            targets = batch["target"]

            noisy = noisy.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if use_spkid:
                spk_id = batch["spk"].to(device, non_blocking=True)

            if noisy.shape[-1] == 0 or targets.shape[-1] == 0:
                logging.warning(
                    f"Bad audio! {batch['id']} noisy: {noisy.shape} target: {targets.shape}"
                )
                continue

            if do_stft:
                noisy = stft(noisy)
                if use_spkid:
                    spk_id = stft(spk_id)
                    batch["spk_lens"] = (
                        batch["spk_lens"] - stft.n_fft
                    ) // stft.hop_length

            optimizer.zero_grad()

            if use_spkid:
                processed = model(noisy, spk_id, batch["spk_lens"]).squeeze(1)
            else:
                processed = model(noisy)

            if do_stft:
                processed = stft.inverse(processed)
                noisy = stft.inverse(noisy)

            processed, targets, use_val = check_lengths(
                batch["id"], processed, targets, "train", do_stft
            )
            if not use_val:
                continue

            loss = loss_fn(processed, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            gromit.train_loss.update(loss.detach())

            save_sample(
                model_cfg.input.sample_rate,
                processed,
                batch["id"],
                trainsaves,
                "train",
                epoch,
                noisy,
                targets,
                gromit,
            )

        do_checkpoint = epoch % ckpt_interval == 0

        if do_checkpoint:
            model.eval()
            if debug:
                loader = tqdm(devset, desc="Validation loop")
            else:
                loader = devset

            with torch.no_grad():
                for batch in loader:

                    noisy = batch["noisy"]
                    targets = batch["target"]

                    noisy = noisy.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)

                    if use_spkid:
                        spk_id = batch["spk"].to(device, non_blocking=True)

                    if noisy.shape[-1] == 0 or targets.shape[-1] == 0:
                        logging.warning(
                            f"Bad audio! {batch['id']} noisy: {noisy.shape} target: {targets.shape}"
                        )
                        continue

                    if do_stft:
                        noisy = stft(noisy)
                        if use_spkid:
                            spk_id = stft(spk_id)
                            batch["spk_lens"] = (
                                batch["spk_lens"] - stft.n_fft
                            ) // stft.hop_length

                    if use_spkid:
                        processed = model(noisy, spk_id, batch["spk_lens"]).squeeze(1)
                    else:
                        processed = model(noisy)

                    if do_stft:
                        processed = stft.inverse(processed)
                        noisy = stft.inverse(noisy)

                    processed, targets, use_val = check_lengths(
                        batch["id"], processed, targets, "val", do_stft
                    )

                    loss = loss_fn(processed, targets)
                    stoi_processed = processed.reshape(-1, processed.shape[-1])
                    for length, proc, targ in zip(
                        batch["noisy_lens"], stoi_processed, targets
                    ):
                        stoi_score = stoi_fn(
                            proc[:length].unsqueeze(0), targ[:length].unsqueeze(0)
                        )
                        gromit.val_stoi.update(-stoi_score[0])

                    gromit.val_loss.update(loss.detach())

                    save_sample(
                        model_cfg.input.sample_rate,
                        processed,
                        batch["id"],
                        devsaves,
                        "dev",
                        epoch,
                        noisy,
                        targets,
                        gromit,
                    )

        gromit.epoch_report(epoch, do_checkpoint, model)
