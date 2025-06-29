import torch
import soxr

from inference.registry import register_enhancement


@register_enhancement("passthrough")
def process_session(
    noisy_audio: torch.Tensor,
    noisy_fs: int,
    spkid_audio: torch.Tensor,
    spkid_fs: int,
    target_fs: int,
):
    output = soxr.resample(noisy_audio[0].detach().cpu().numpy(), noisy_fs, target_fs)
    return torch.from_numpy(output).unsqueeze(0)
