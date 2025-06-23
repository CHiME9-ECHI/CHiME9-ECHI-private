"""Prepare ECHI submission for evaluation"""

import logging

import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from signal_tools import get_session_tuples, segment_signal

# from signal_tools import get_session_tuples


def prepare(cfg):
    logging.info("Running preparation for evaluation")

    session_tuples = get_session_tuples(
        cfg.sessions_file, cfg.devices, datasets=[cfg.dataset]
    )

    output_dir_template = cfg.segment_dir
    signal_template = cfg.enhanced_signal

    for session, device, pid in tqdm(session_tuples):
        # Segment the reference signal for this PID
        output_dir = output_dir_template.format(
            dataset=cfg.dataset, device=device, segment_type="individual"
        )

        logging.info(f"Segmenting {device}, {pid} signals into {output_dir}")
        wav_file = signal_template.format(
            dataset=cfg.dataset, session=session, device=device, pid=pid
        )
        csv_file = cfg.segment_info_file.format(
            dataset=cfg.dataset, session=session, device=device, pid=pid
        )
        segment_signal(wav_file, csv_file, output_dir)

        # Segment the summed reference signal using this PIDs segment info
        output_dir = output_dir_template.format(
            dataset=cfg.dataset, device=device, segment_type="summed"
        )
        logging.info(f"Segmenting {device}, {pid} reference signals into {output_dir}")

        pids = [p for s, d, p in session_tuples if s == session and d == device]
        wav_files = [
            signal_template.format(
                dataset=cfg.dataset, session=session, device=device, pid=pid
            )
            for pid in pids
        ]

        segment_signal(wav_files, csv_file, output_dir)


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    prepare(cfg.prepare)


if __name__ == "__main__":
    main()
