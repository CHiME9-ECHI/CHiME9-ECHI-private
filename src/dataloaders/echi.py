if __name__ == "__main__":
    import sys

    sys.path.append("src")

import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd

from utils.audio_prep import AudioPrep
from utils.signal_utils import combine_audio_list


def collate_fn(batch: list[dict]):
    new_out = {"id": [x["id"] for x in batch]}

    for audio_type in ["noisy", "target", "spk"]:
        audio = [x[audio_type] for x in batch]
        audio, lens = combine_audio_list(audio)
        new_out[audio_type] = audio
        new_out[audio_type + "_lens"] = lens

    return new_out


class ECHI(Dataset):
    def __init__(
        self,
        name,
        metadata,
        audio_dir,
        audio_device,
        data_split,
        onesession,
        debug,
        noisy_prep: AudioPrep,
        ref_prep: AudioPrep,
        spk_prep: AudioPrep,
    ):
        super().__init__()
        self.name = name
        self.metadata = pd.read_csv(metadata.format(subset=data_split)).dropna(axis=0)
        self.metadata = self.metadata.to_dict(orient="records")
        self.audio_device = audio_device

        if onesession:
            target = "train_14" if data_split == "train" else "dev_02"
            self.metadata = [x for x in self.metadata if x["session"] == target]

        self.device_dir = (
            Path(audio_dir) / f"{self.audio_device}/{data_split}_16k_speech"
        )
        self.ref_dir = Path(audio_dir) / f"ref/{data_split}_16k_speech"
        self.spkid_dir = Path(audio_dir) / f"participant/{data_split}_16k"
        self.segment_dir = Path(audio_dir) / f"metadata/ref/{data_split}"

        self.segment_samples = 16000 * 4
        self.trainset = data_split == "train"

        self.device_filestring = "{session}/{session}.{device}.{pid}.{segment}.wav"
        self.ref_filestring = "{session}/{session}.{device}_ref.{pid}.{segment}.wav"
        self.segments_filestring = "{session}.{device}.{pid}.csv"

        self.preppers = {"noisy": noisy_prep, "target": ref_prep, "spk": spk_prep}

        self.manifest: list[dict]
        self.make_manifest()

        self.debug = debug
        if self.debug:
            self.manifest = self.manifest[:10]

    def make_manifest(self):
        self.manifest = []

        for meta in self.metadata:
            device_pos = int(meta[f"{self.audio_device}_pos"])
            session = meta["session"]
            pids = [meta[f"pos{i}"] for i in range(1, 5) if i != device_pos]

            if not self.check_spkids(pids):
                # Ignore whole session if spkid passage doesn't exist
                continue
            for pid in pids:
                segment_fpath = self.segment_dir / self.segments_filestring.format(
                    session=session, device=self.audio_device, pid=pid
                )
                if not segment_fpath.exists():
                    continue
                segments = pd.read_csv(
                    self.segment_dir
                    / self.segments_filestring.format(
                        session=meta["session"], device=self.audio_device, pid=pid
                    ),
                    header=None,
                ).values.tolist()
                for seg_id, start, end in segments:
                    seg_id = str(seg_id).zfill(3)
                    if not self.check_segments(session, pid, seg_id):
                        continue
                    elif (end - start) / 16000 < 1:
                        continue
                    self.manifest.append(
                        {
                            "id": f"{session}.{pid}.{seg_id}",
                            "noisy": self.get_device_fpath(session, pid, seg_id),
                            "target": self.get_ref_fpath(session, pid, seg_id),
                            "spk": self.get_spkid_path(pid),
                        }
                    )

    def __getitem__(self, index):
        meta = self.manifest[index]

        out = {"id": meta["id"]}

        for audio_type in ["noisy", "target", "spk"]:
            audio, fs = torchaudio.load(str(meta[audio_type]))
            prep = self.preppers[audio_type]  # type: AudioPrep
            audio = prep.process(audio, fs)

            if (
                audio.shape[-1] > self.segment_samples
                and audio_type != "spk"
                and self.trainset
            ):
                audio = audio[..., : self.segment_samples]

            out[audio_type] = audio

        return out

    def __len__(self):
        return len(self.manifest)

    def get_device_fpath(self, sess_id, pid, seg_id):
        return self.device_dir / self.device_filestring.format(
            session=sess_id, device=self.audio_device, pid=pid, segment=seg_id
        )

    def get_ref_fpath(self, sess_id, pid, seg_id):
        return self.ref_dir / self.ref_filestring.format(
            session=sess_id, device=self.audio_device, pid=pid, segment=seg_id
        )

    def get_spkid_path(self, pid):
        return self.spkid_dir / f"{pid}.wav"

    def check_spkids(self, pids):
        good = True
        for pid in pids:
            if not self.get_spkid_path(pid).exists():
                good = False
                break
        return good

    def check_segments(self, sess_id, pid, seg_id):
        device = self.get_device_fpath(sess_id, pid, seg_id).exists()
        ref = self.get_ref_fpath(sess_id, pid, seg_id).exists()
        return device and ref
