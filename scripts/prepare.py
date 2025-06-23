"""Prepare ECHI data"""

import logging

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from signal_tools import get_session_tuples, segment_signal


def prepare(cfg):
    logging.info("Preparing the ECHI dataset")

    session_tuples = get_session_tuples(
        cfg.sessions_file, cfg.devices, datasets=[cfg.dataset]
    )

    for session, device, pid in tqdm(session_tuples):
        # for device in cfg.devices:
        output_dir = cfg.ref_segment_dir.format(dataset=cfg.dataset, device=device)

        logging.info(f"Segmenting {device} reference signals into {output_dir}")

        wav_file = cfg.ref_signal_file.format(
            dataset=cfg.dataset, session=session, device=device, pid=pid
        )
        csv_file = cfg.segment_info_file.format(
            dataset=cfg.dataset, session=session, device=device, pid=pid
        )
        segment_signal(wav_file, csv_file, output_dir)


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    prepare(cfg.prepare)


if __name__ == "__main__":
    main()
