import torchaudio
from pathlib import Path
from torch.utils.data import Dataset


class ECHI_Inference(Dataset):
    def __init__(
        self,
        name,
        audio_dir,
        session,
        target_pids,
        audio_device,
        data_split,
        sample_rate,
        window_len,
        stride,
    ):
        super().__init__()

        self.name = name

        self.audio_dir = Path(audio_dir)
        self.audio_device = audio_device

        self.data_split = data_split
        self.session = session
        self.target_pids = target_pids

        self.sample_rate = sample_rate
        self.window_len = window_len
        self.stride = stride

        self.device_audio_dir = (
            self.audio_dir
            / self.audio_device
            / f"{data_split}_{self.sample_rate // 1000}k_window{self.window_len}s_stride{self.stride}s"
            / self.session
        )
        self.participant_dir = (
            self.audio_dir / "participant" / f"{data_split}_{self.sample_rate // 1000}k"
        )

        self.make_manifest()
        self.manifest = self.manifest[:12]

    def make_manifest(self):
        self.manifest = []

        for file in self.device_audio_dir.glob("*.wav"):
            name = file.name

            for pid in self.target_pids:
                self.manifest.append(
                    {
                        "id": name,
                        "noisy_fpath": file,
                        "participant": pid,
                        "spk_fpath": self.participant_dir / f"{pid}.wav",
                    }
                )

    def __getitem__(self, index):
        info = self.manifest[index]

        noisy, fs = torchaudio.load(info["noisy_fpath"])
        assert fs == self.sample_rate

        spk, fs = torchaudio.load(info["spk_fpath"])
        assert fs == self.sample_rate

        output = {
            "id": info["id"],
            "participant": info["participant"],
            "noisy": noisy,
            "spk": spk,
            "spk_lens": spk.shape[-1],
        }

        return output

    def __len__(self):
        return len(self.manifest)
