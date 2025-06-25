"""Validation script for ECHI submission."""

import logging
from pathlib import Path

import hydra
import soundfile as sf
from omegaconf import DictConfig

from evaluation.signal_tools import get_session_tuples

EXPECTED_SAMPLE_RATE = 16000
EXPECTED_FORMAT = "PCM_16"
EXPECTED_N_CHANNELS = 1


def validate_signal(
    dataset, session, device, pid, noisy_signal_template, enhanced_signal_template
):
    """Validate a single signal for ECHI submission."""
    logging.debug(
        f"Validating signal for session {session}, device {device}, pid {pid}"
    )

    errors = []

    noisy_signal_file = noisy_signal_template.format(
        dataset=dataset, session=session, device=device, pid=pid
    )
    enhanced_signal_file = enhanced_signal_template.format(
        dataset=dataset, session=session, device=device, pid=pid
    )
    logging.debug(f"Checking: {enhanced_signal_file} against {noisy_signal_file}")

    # This test should pass if CHiME9-ECHI data is installed
    if not Path(noisy_signal_file).exists():
        errors.append(f"Noisy signal file does not exist: {noisy_signal_file}")
        errors.append("There is an issue with your CHiME9-ECHI data installation.")
        errors.append("Fix the issue before proceeding with validation.")
        return errors

    # Check that the enhanced signal file exists
    if not Path(enhanced_signal_file).exists():
        error_msg = f"Enhanced signal file does not exist: {enhanced_signal_file}"
        errors.append(error_msg)

    try:
        noisy_signal_info = sf.info(noisy_signal_file)
    except RuntimeError as e:
        errors.append(f"Error reading noisy signal file {noisy_signal_file}: {e}")
        return errors
    try:
        enhanced_signal_info = sf.info(enhanced_signal_file)
    except RuntimeError as e:
        error_msg = f"Error reading enhanced signal file {enhanced_signal_file}: {e}"
        errors.append(error_msg)
        return errors

    # Check the enhanced signal is at least as long as the noisy signal
    if enhanced_signal_info.duration < noisy_signal_info.duration:
        error_msg = (
            f"Enhanced signal {enhanced_signal_file} is shorter than noisy signal "
            f"{noisy_signal_file}. Duration: {enhanced_signal_info.duration} < "
            f"{noisy_signal_info.duration}."
        )
        errors.append(error_msg)

    # Check that the sample rate is 16 kHz for the enhanced signal
    if enhanced_signal_info.samplerate != EXPECTED_SAMPLE_RATE:
        error_msg = (
            f"Enhanced signal {enhanced_signal_file} has incorrect sample rate: "
            f"Observed {enhanced_signal_info.samplerate}; "
            f"Expected {EXPECTED_SAMPLE_RATE}."
        )
        errors.append(error_msg)

    # Check that the submitted signal is 16-bit PCM
    if enhanced_signal_info.subtype != EXPECTED_FORMAT:
        error_msg = (
            f"Enhanced signal {enhanced_signal_file} is not {EXPECTED_FORMAT} format. "
            f"Observed {enhanced_signal_info.subtype}; Expected {EXPECTED_FORMAT}."
        )
        errors.append(error_msg)

    # Check that the number of channels is 1 for the enhanced signal
    if enhanced_signal_info.channels != EXPECTED_N_CHANNELS:
        error_msg = (
            f"Enhanced signal {enhanced_signal_file} has incorrect number of channels: "
            f"Observed {enhanced_signal_info.channels}; Expected {EXPECTED_N_CHANNELS}."
        )
        errors.append(error_msg)

    return errors


def validate(cfg):
    """Iterate over all expected signals to check for errors"""
    logging.info("Validating ECHI submission")
    is_valid = True
    session_tuples = get_session_tuples(
        cfg.sessions_file, cfg.devices, datasets=[cfg.dataset]
    )

    errors = []
    for session, device, pid in session_tuples:
        errors.extend(
            validate_signal(
                dataset=cfg.dataset,
                session=session,
                device=device,
                pid=pid,
                noisy_signal_template=cfg.noisy_signal,
                enhanced_signal_template=cfg.enhanced_signal,
            )
        )

    is_valid = len(errors) == 0
    if not is_valid:
        logging.error("Validation failed with the following errors:")
        for error in errors:
            logging.error(error)
        logging.error("Validation FAILED. Fix your submission before proceeding.")
    else:
        logging.info("Validation passed successfully.")
    return is_valid


@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig) -> None:
    if not validate(cfg.validate):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
