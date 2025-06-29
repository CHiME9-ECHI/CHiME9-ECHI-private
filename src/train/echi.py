if __name__ == "__main__":
    import sys

    sys.path.append("src")

import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
import csv

from shared.signal_utils import AudioPrep, combine_audio_list


from typing import Any


def collate_fn(batch: list[dict[str, Any]]):
    new_out: dict[str, Any] = {"id": [x["id"] for x in batch]}

    for audio_type in ["noisy", "target", "spkid"]:
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
        noisy_signal: str,
        ref_signal: str,
        rainbow_signal: str,
        sessions_file: str,
        segments_file: str,
        debug: bool,
        input_prepper: AudioPrep,
    ):
        super().__init__()
        self.subset = subset
        self.audio_device = audio_device

        with open(sessions_file.format(dataset=subset), "r") as f:
            self.metadata = list(csv.DictReader(f))

        self.segments_file = segments_file

        self.signal_paths = {
            "noisy": noisy_signal,
            "target": ref_signal,
            "spkid": rainbow_signal,
        }

        self.segment_samples = 16000 * 4

        self.prepper = input_prepper
        self.noisy_channels = input_prepper.output_channels

        self.debug = debug

        self.manifest: list[dict]
        self.make_manifest()

    def make_manifest(self):
        self.manifest = []

        end = False

        for meta in self.metadata:

            try:
                device_pos = int(meta[f"{self.audio_device}_pos"])
            except ValueError:
                continue
            pids = [meta[f"pos{i}"] for i in range(1, 5) if i != device_pos]

            for pid in pids:
                with open(
                    self.segments_file.format(
                        dataset=self.subset,
                        session=meta["session"],
                        device=self.audio_device,
                        pid=pid,
                    ),
                    "r",
                ) as f:
                    segments = list(
                        csv.DictReader(f, fieldnames=["index", "start", "end"])
                    )

                self.manifest += self.get_segment_paths(meta["session"], pid, segments)

                if self.debug and len(self.manifest) > 10:
                    self.manifest = self.manifest[:10]
                    end = True
                    break
            if end:
                break

    def get_segment_paths(self, session, pid, segments) -> list[dict]:

        good_files = []
        for seg in segments:
            all_good = True
            seg_fpaths = {}
            for audio_type, fpath in self.signal_paths.items():
                this_fpath = fpath.format(
                    dataset=self.subset,
                    session=session,
                    device=self.audio_device,
                    pid=pid,
                    segment=str(seg["index"]).zfill(3),
                )

                if not Path(this_fpath).exists():
                    all_good = False
                    break

                seg_fpaths[audio_type] = this_fpath
            if all_good:
                length = (int(seg["end"]) - int(seg["start"])) / 16000
                if length < 1:
                    continue

                seg_fpaths["id"] = seg_fpaths["noisy"].split("/")[-1][:-4]
                good_files.append(seg_fpaths)

        return good_files

    def __getitem__(self, index):
        meta = self.manifest[index]

        out = {"id": meta["id"]}

        for audio_type in self.signal_paths.keys():
            audio, fs = torchaudio.load(str(meta[audio_type]))

            if audio_type == "noisy":
                self.prepper.output_channels = self.noisy_channels
            else:
                self.prepper.output_channels = 1

            audio = self.prepper.process(audio, fs)

            if audio.shape[-1] > self.segment_samples and audio_type != "spk":
                # Cut segments short to avoid memory issues
                audio = audio[..., : self.segment_samples]

            out[audio_type] = audio

        self.prepper.output_channels = self.noisy_channels
        return out

    def __len__(self):
        return len(self.manifest)
