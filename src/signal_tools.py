"""Tools for handling signals"""

import csv
import itertools
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

POSITIONS = ["pos1", "pos2", "pos3", "pos4"]


def get_session_tuples(session_file, devices, datasets=None):
    """Get session tuples for the specified datasets and devices."""
    with open(session_file, "r") as f:
        sessions = list(csv.DictReader(f))

    # Filter sessions for the specified datasets
    if datasets is not None:
        sessions = [s for s in sessions if s["session"].startswith(tuple(datasets))]
    session_device_pid_tuples = []

    for device, session in itertools.product(devices, sessions):
        device_pos = "pos" + session[f"{device}_pos"]
        pids = [session[pos] for pos in POSITIONS if pos != device_pos]
        for pid in pids:
            session_device_pid_tuples.append((session["session"], device, pid))

    return session_device_pid_tuples


def read_wav_files_and_sum(wav_files):
    """Read a list of wav files and return their sum."""

    sum_signal = None
    fs_set = set()
    for file in wav_files:
        with open(file, "rb") as f:
            signal, fs = sf.read(f)
            fs_set.add(fs)
            if sum_signal is not None:
                if len(signal) != len(sum_signal):
                    # pad the short with zeros
                    if len(signal) < len(sum_signal):
                        signal = np.pad(signal, (0, len(sum_signal) - len(signal)))
                    else:
                        sum_signal = np.pad(
                            sum_signal, (0, len(signal) - len(sum_signal))
                        )
                sum_signal += signal
            else:
                sum_signal = signal
    if len(fs_set) != 1:
        raise ValueError(f"Inconsistent sampling rates found: {fs_set}")
    fs = fs_set.pop()

    return sum_signal, fs


def wav_file_name(
    output_dir: Path, stem: str, index: int, start_sample: int, end_sample: int
) -> Path:
    """Construct the wav file name based on session, device, and pid."""
    return Path(output_dir) / f"{stem}.{index:03g}.wav"
    # return Path(output_dir) / f"{stem}.{index:03g}.{start_sample}_{end_sample}.wav"


def segment_signal(
    wav_file: Path | list[Path], csv_file: Path, output_dir: Path
) -> None:
    """Extract speech segments from a signal"""
    logging.debug(f"Segmenting {wav_file} {csv_file}")
    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(csv_file, "r") as f:
        segments = list(csv.DictReader(f, fieldnames=["index", "start", "end"]))

    # check if any files missing:
    files_missing = False
    for segment in segments:
        expected_files = wav_file_name(
            output_dir,
            Path(csv_file).stem,
            int(segment["index"]),
            int(segment["start"]),
            int(segment["end"]),
        )
        if not expected_files.exists():
            files_missing = True
            break
    if not files_missing:
        logging.debug(f"All segments already exist in {output_dir}")
        return

    if isinstance(wav_file, list):
        signal, fs = read_wav_files_and_sum(wav_file)
    else:
        with open(wav_file, "rb") as f:
            signal, fs = sf.read(f)

    logging.debug(f"Will generate {len(segments)} segments from {wav_file}")
    # sample_scalar = fs / seg_sample_rate
    # collar_samples = fs * collar_ms
    sample_scalar = 1
    collar_samples = 0

    for segment in segments:
        index = int(segment["index"])
        start_sample = int(int(segment["start"]) * sample_scalar) - collar_samples
        end_sample = int(int(segment["end"]) * sample_scalar) + collar_samples

        output_file = wav_file_name(
            output_dir, Path(csv_file).stem, index, start_sample, end_sample
        )
        if output_file.exists():
            logging.debug(f"Segment {output_file} already exists, skipping")
            continue
        if end_sample > len(signal):
            logging.warning(f"Segment {output_file} exceeds signal length. Skipping.")
            continue
        signal_segment = signal[start_sample:end_sample]
        with open(output_file, "wb") as f:
            sf.write(f, signal_segment, samplerate=fs)


def csv_to_pid_wav(name: str) -> str:
    """Replace .wav with .csv"""
    return ".".join(name.split(".")[:-1]) + ".csv"


def segment_all_signals(
    signal_template, output_dir_template, segment_info_file, dataset, session_tuples
):
    for session, device, pid in tqdm(session_tuples):
        # Segment the reference signal for this PID
        output_dir = output_dir_template.format(
            dataset=dataset, device=device, segment_type="individual"
        )

        logging.info(f"Segmenting {device}, {pid} reference signals into {output_dir}")
        wav_file = signal_template.format(
            dataset=dataset, session=session, device=device, pid=pid
        )
        csv_file = segment_info_file.format(
            dataset=dataset, session=session, device=device, pid=pid
        )
        segment_signal(wav_file, csv_file, output_dir)

        # Segment the summed reference signal using this PIDs segment info
        output_dir = output_dir_template.format(
            dataset=dataset, device=device, segment_type="summed"
        )
        logging.info(f"Segmenting {device}, {pid} reference signals into {output_dir}")

        pids = [p for s, d, p in session_tuples if s == session and d == device]
        wav_files = [
            signal_template.format(
                dataset=dataset, session=session, device=device, pid=pid
            )
            for pid in pids
        ]

        segment_signal(wav_files, csv_file, output_dir)


### Function below is OBSOLETE and marked for removal before release.


# def segment_signal_dir(
#     signal_dir: Path | str,
#     segment_info_dir: Path | str,
#     output_dir: Path | str,
#     segment_sample_rate: int,
#     segment_collar: int,
#     file_pattern: str = "*",
#     translate_id: Optional[str] = None,
# ) -> None:
#     """Extract speech segments from all signals in a directory"""
#     logging.info("Segmenting signals...")
#     if translate_id == "pid_wav":
#         translate_fn = csv_to_pid_wav
#     elif translate_id == "device_wav":
#         translate_fn = csv_to_device_wav

#     output_dir = Path(output_dir)
#     if not output_dir.exists():
#         output_dir.mkdir(parents=True, exist_ok=True)

#     # Find all the csv segmentation files to process...
#     csv_files = list(Path(segment_info_dir).glob(file_pattern))
#     # ... and find their corresponding wav files
#     wav_files = [
#         Path(signal_dir) / translate_fn(str(csv_file.name)) for csv_file in csv_files
#     ]

#     n_files = len(wav_files)

#     for wav_file, csv_file in tqdm(
#         zip(wav_files, csv_files), desc="Segmenting...", total=n_files
#     ):
#         if not wav_file.exists():
#             logging.error(f"Missing wav file: {wav_file}")
#             continue
#         segment_signal(
#             wav_file, csv_file, Path(output_dir), segment_sample_rate, segment_collar
#         )
