"""Tools for handling signals"""

import csv
import itertools
import logging
from pathlib import Path
from typing import Optional

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


def wav_file_name(
    output_dir: Path, stem: str, index: int, start_sample: int, end_sample: int
) -> Path:
    """Construct the wav file name based on session, device, and pid."""
    return Path(output_dir) / f"{stem}.{index:03g}.wav"
    # return Path(output_dir) / f"{stem}.{index:03g}.{start_sample}_{end_sample}.wav"


def segment_signal(
    wav_file: Path, csv_file: Path, output_dir: Path, seg_sample_rate
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
            csv_file.stem,
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

    with open(wav_file, "rb") as f:
        signal, fs = sf.read(f)

    sample_scalar = fs / seg_sample_rate

    for segment in segments:
        index = int(segment["index"])
        start_sample = int(int(segment["start"]) * sample_scalar)
        end_sample = int(int(segment["end"]) * sample_scalar)

        output_file = wav_file_name(
            output_dir, csv_file.stem, index, start_sample, end_sample
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
    return ".".join(name.split(".")[:-1]) + ".wav"


def csv_to_device_wav(name: str) -> str:
    """Replace PXXX.csv with .wav"""
    return ".".join(name.split(".")[:-2]) + ".wav"


def segment_signal_dir(
    signal_dir: Path | str,
    segment_info_dir: Path | str,
    output_dir: Path | str,
    segment_sample_rate: int,
    file_pattern: str = "*",
    translate_id: Optional[str] = None,
) -> None:
    """Extract speech segments from all signals in a directory"""
    logging.info("Segmenting signals...")
    if translate_id == "pid_wav":
        translate_fn = csv_to_pid_wav
    elif translate_id == "device_wav":
        translate_fn = csv_to_device_wav

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # Find all the csv segmentation files to process...
    csv_files = list(Path(segment_info_dir).glob(file_pattern))
    # ... and find their corresponding wav files
    wav_files = [
        Path(signal_dir) / translate_fn(str(csv_file.name)) for csv_file in csv_files
    ]

    n_files = len(wav_files)

    for wav_file, csv_file in tqdm(
        zip(wav_files, csv_files), desc="Segmenting...", total=n_files
    ):
        if not wav_file.exists():
            logging.error(f"Missing wav file: {wav_file}")
            continue
        segment_signal(wav_file, csv_file, Path(output_dir), segment_sample_rate)
