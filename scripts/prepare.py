"""Prepaer ECHI submission for evaluation"""

import logging

import hydra
from omegaconf import DictConfig

# from signal_tools import get_session_tuples


def prepare(cfg):
    logging.info("Running preparation for evaluation")
    # TODO: Complete this
    # signal_dir = cfg.enhanced_dir

    for device in cfg.devices:
        # session_device_pid_tuples = get_session_tuples(
        #     cfg.sessions_file, [device], datasets=[cfg.dataset]
        # )

        segment_dir = cfg.segment_dir.format(device=device, segment_type="individual")
        logging.info(f"Segment {device} signals into {segment_dir}")
        # segment_signal_dir(signal_dir, cfg.csv_dir, segment_dir, filter=f"*{device}*P*")


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    prepare(cfg.evaluate)


if __name__ == "__main__":
    main()
