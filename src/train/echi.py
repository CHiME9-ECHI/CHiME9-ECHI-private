if __name__ == "__main__":
    import sys

    sys.path.append("src")

import soundfile as sf
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd

from evaluation.signal_tools import AudioPrep, combine_audio_list


from typing import Any


def collate_fn(batch: list[dict[str, Any]]):
    new_out: dict[str, Any] = {"id": [x["id"] for x in batch]}

    for audio_type in ["noisy", "target", "spk"]:
        audio = [x[audio_type] for x in batch]
        audio, lens = combine_audio_list(audio)
        new_out[audio_type] = audio
        new_out[audio_type + "_lens"] = lens

    return new_out


class ECHI(Dataset):
    def __init__(
        self,
        subset: str,
        audio_device: str,
        noisy_dir: str,
        ref_dir: str,
        rainbow_dir: str,
        sessions_file: str,
        debug: bool,
        noisy_prep: AudioPrep,
        ref_prep: AudioPrep,
        spk_prep: AudioPrep,
    ):
        super().__init__()
        self.subset = subset
        self.audio_device = audio_device

        self.metadata = (
            pd.read_csv(sessions_file.format(dataset=subset))
            .dropna(axis=0)
            .to_dict(orient="records")
        )  # type: list[dict]

        self.noisy_dir = Path(noisy_dir.format(device=audio_device, dataset=subset))
        self.ref_dir = Path(ref_dir.format(device=audio_device, dataset=subset))
        self.spkid_dir = Path(rainbow_dir.format(dataset=subset))

        self.segment_samples = 16000 * 4

        self.preppers = {"noisy": noisy_prep, "target": ref_prep, "spk": spk_prep}

        self.manifest: list[dict]
        self.make_manifest()

        self.debug = debug
        if self.debug:
            self.manifest = self.manifest[:10]

    def make_manifest(self):
        self.manifest = []

        session_wearer_pids = {}
        for meta in self.metadata:
            device_pos = int(meta[f"{self.audio_device}_pos"])
            session_wearer_pids[meta["session"]] = meta[f"pos{device_pos}"]

        noisy_files = self.noisy_dir.glob("*")
        for noisy in noisy_files:
            session, _, pid, _, _ = noisy.name.split(".")
            if pid == session_wearer_pids[session]:
                # Skip files where the speech is from the device wearer
                continue
            ref = self.ref_dir / noisy.name
            rainbow = self.spkid_dir / f"{pid}.wav"

            if not ref.exists() or not rainbow.exists():
                # print(ref, ref.exists())
                # print(rainbow, rainbow.exists())
                continue

            with sf.SoundFile(str(noisy)) as file:
                dur = file.frames / file.samplerate
            if dur < 1:
                continue

            self.manifest.append(
                {"id": noisy.name, "noisy": noisy, "target": ref, "spk": rainbow}
            )

    def __getitem__(self, index):
        meta = self.manifest[index]

        out = {"id": meta["id"]}

        for audio_type in ["noisy", "target", "spk"]:
            audio, fs = torchaudio.load(str(meta[audio_type]))
            prep = self.preppers[audio_type]  # type: AudioPrep
            audio = prep.process(audio, fs)

            if audio.shape[-1] > self.segment_samples and audio_type != "spk":
                # Cut segments short to avoid memory issues
                audio = audio[..., : self.segment_samples]

            out[audio_type] = audio

        return out

    def __len__(self):
        return len(self.manifest)
