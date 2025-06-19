import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path

from utils.file_utils import read_json
from utils.signal_utils import match_length, pad_tolength
from utils.audio_prep import AudioPrep


def collate_fn(batch: list[dict]):
    output = {"id": [x["id"] for x in batch]}

    for key in ["noisy", "target", "spk"]:
        audio = [x[key] for x in batch]
        lens = [a.shape[-1] for a in audio]
        if len(set(lens)) == 1:
            output[key] = torch.stack(audio)
            output[f"{key}_lens"] = torch.tensor(lens, dtype=int)
        else:
            max_len = max(lens)
            new_audio = []
            for x in audio:
                new_audio.append(pad_tolength(x, max_len - x.shape[-1]))
            output[key] = torch.stack(new_audio)
            output[f"{key}_lens"] = torch.tensor(
                [max_len for _ in new_audio], dtype=int
            )
    return output


class ClarityData(Dataset):
    def __init__(
        self,
        name: str,
        metadata_path: str,
        audio_dir: str,
        spk_dir: str,
        load_sr: int,
        segment_length: int,
        split: str,
        debug: bool,
        noisy_prep: AudioPrep,
        ref_prep: AudioPrep,
    ):
        self.name = name
        self.split = split
        self.debug = debug

        self.audio_root = Path(audio_dir.format(subset=split))
        self.metadata = read_json(Path(metadata_path.format(subset=split)))
        self.spk_dir = Path(spk_dir.format(subset=split))

        self.noisy_prep = noisy_prep
        self.ref_prep = ref_prep

        self.load_sr = load_sr
        self.segment_samples = segment_length * self.noisy_prep.output_sr
        self.spk_samples = 20 * 16000

        self.mixed_suffix = [f"_mix_CH{i}.wav" for i in range(1, 4)]
        self.target_suffix = "_target_anechoic_CH1.wav"

        self.manifest: list[dict]
        self.make_manifest()

        if self.debug:
            self.manifest = self.manifest[:10]

    def make_manifest(self):
        self.manifest = []  # type: list[dict]

        for meta in self.metadata:
            if not self.check_exists(meta["scene"]):
                continue

            thing = meta
            thing["ha_paths"] = self.get_ha_paths(meta["scene"])
            thing["ref_path"] = self.get_ref_path(meta["scene"])

            target_spk = thing["target"]["name"].split("_")[0]
            thing["spk_files"] = self.get_spk_paths(target_spk)

            self.manifest.append(thing)

    def __getitem__(self, index):
        meta = self.manifest[index]

        ha_audio = []
        for ha_path in meta["ha_paths"]:
            audio, fs = torchaudio.load(ha_path)
            assert (
                fs == self.load_sr
            ), f"HA audio has wrong sample rate. Expected {self.load_sr}Hz, found {fs}Hz"
            ha_audio.append(audio)
        ha_audio = torch.cat(ha_audio, dim=0)
        ha_audio = self.noisy_prep.process(ha_audio, self.load_sr)

        ref_audio, fs = torchaudio.load(meta["ref_path"])
        assert (
            fs == self.load_sr
        ), f"Ref audio has wrong sample rate. Expected {self.load_sr}Hz, found {fs}Hz"

        ref_audio = self.ref_prep.process(ref_audio, self.load_sr)

        ha_audio, ref_audio = match_length(ha_audio, ref_audio)
        if self.segment_samples > 0:
            ha_audio, ref_audio = self.get_audio_segment(ha_audio, ref_audio)

        spk_files = []
        for fpath in meta["spk_files"]:
            audio, fs = torchaudio.load(fpath)
            assert fs == self.load_sr
            spk_files.append(audio)

        spk_files = torch.cat(spk_files, dim=-1)
        spk_files = self.ref_prep.process(spk_files, fs)

        if spk_files.shape[-1] > self.spk_samples:
            spk_files = spk_files[:, : self.spk_samples]
        elif spk_files.shape[-1] < self.spk_samples:
            pad_len = self.spk_samples - spk_files.shape[-1]
            spk_files = torch.nn.functional.pad(spk_files, (0, pad_len))

        output = {
            "id": meta["scene"],
            "noisy": ha_audio,
            "target": ref_audio,
            "spk": spk_files,
        }
        return output

    def __len__(self):
        return len(self.manifest)

    def get_audio_segment(self, ha_audio: torch.Tensor, ref_audio: torch.Tensor):
        if self.segment_samples > ha_audio.shape[-1]:
            pad_len = self.segment_samples - ha_audio.shape[-1]
            ha_audio = torch.nn.functional.pad(ha_audio, (0, pad_len))
            ref_audio = torch.nn.functional.pad(ref_audio, (0, pad_len))

            return ha_audio, ref_audio

        if ha_audio.shape[-1] == self.segment_samples:
            return ha_audio, ref_audio

        max_start = ha_audio.shape[-1] - self.segment_samples
        start = torch.randint(0, max_start, (1,))
        end = start + self.segment_samples
        return ha_audio[..., start:end], ref_audio[..., start:end]

    def get_ha_paths(self, scene):
        return [self.audio_root / f"{scene}{suff}" for suff in self.mixed_suffix]

    def get_ref_path(self, scene):
        return self.audio_root / f"{scene}{self.target_suffix}"

    def get_spk_paths(self, speaker):
        files = self.spk_dir.glob(f"{speaker}*.wav")
        files = list(files)[:4]
        return files

    def check_exists(self, scene):
        fpaths = self.get_ha_paths(scene)
        fpaths.append(self.get_ref_path(scene))
        exists = True
        for fpath in fpaths:
            if not fpath.exists():
                exists = False
                break
        return exists
