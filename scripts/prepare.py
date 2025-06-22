"""Prepare ECHI data"""

import logging

import hydra
from omegaconf import DictConfig

from signal_tools import segment_signal_dir


def prepare(cfg):
    logging.info("Preparing the ECHI dataset")

    for device in cfg.devices:
        signal_dir = cfg.input_signal_dir.format(dataset=cfg.dataset, device=device)
        output_dir = cfg.output_segment_dir.format(dataset=cfg.dataset, device=device)
        logging.info(f"Segmenting {device} reference signals into {output_dir}")
        segment_info_dir = cfg.segment_info_dir.format(dataset=cfg.dataset)
        segment_signal_dir(
            signal_dir=signal_dir,
            segment_info_dir=segment_info_dir,
            output_dir=output_dir,
            file_pattern=f"*{device}*P*",
            translate_id=cfg.translate_id,
        )


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    prepare(cfg.prepare)


if __name__ == "__main__":
    main()
