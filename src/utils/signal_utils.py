import torch
import torchaudio
from typing import Union
import logging


def read_wavfile(path, expected_sr=False):
    if not isinstance(path, str):
        path = str(path)

    audio, fs = torchaudio.load(path)
    if expected_sr:
        assert expected_sr == fs, f"Expected {expected_sr}Hz sr but found {fs}Hz"
    return audio


def get_rms(signal: torch.Tensor) -> torch.Tensor:
    """
    Calculate the RMS of a signal.
    Args:
        signal (torch.Tensor): The input signal.
    Returns:
        torch.Tensor: The RMS of the signal.
    """
    return torch.sqrt(torch.mean(signal**2))


def rms_normalize(signal: torch.Tensor, target_rms: float) -> torch.Tensor:
    """
    Normalize the RMS of a signal to 1.
    Args:
        signal (torch.Tensor): The input signal.
    Returns:
        torch.Tensor: The normalized signal.
    """
    rms = get_rms(signal)
    if rms > 0:
        return signal * target_rms / rms
    else:
        return signal


def match_length(audio0, audio1):
    """
    Pads the shorter of two audio tensors along the last dimension so that both have the same length.
    Parameters:
        audio0 (torch.Tensor): The first audio tensor.
        audio1 (torch.Tensor): The second audio tensor.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the two audio tensors, both padded to the same length along the last dimension.
    """

    if audio0.shape[-1] > audio1.shape[-1]:
        pad_len = audio0.shape[-1] - audio1.shape[-1]
        audio1 = torch.nn.functional.pad(audio1, (0, pad_len))
    elif audio1.shape[-1] > audio0.shape[-1]:
        pad_len = audio1.shape[-1] - audio0.shape[-1]
        audio0 = torch.nn.functional.pad(audio0, (0, pad_len))
    return audio0, audio1


def pad_samples(audio: torch.Tensor, samples: int):
    """
    Pads the input audio tensor with a specified number of zeros at the end.
    Args:
        audio (torch.Tensor): The input audio tensor to be padded.
        samples (int): The number of zero samples to pad at the end of the audio tensor.
    Returns:
        torch.Tensor: The padded audio tensor. If samples is 0, returns the original tensor.
    """

    if samples == 0:
        return audio
    audio = torch.nn.functional.pad(audio, (0, samples), mode="constant", value=0.0)
    return audio


def pad_tolength(audio: torch.Tensor, target_length: int):
    """
    Pads the input audio tensor to the specified target length.
    If the target length is less than or equal to the current length of the audio tensor,
    the original audio is returned. If the target length is greater, the audio is padded
    with zeros (or as defined by `pad_samples`) to reach the target length.
    Args:
        audio (torch.Tensor): The input audio tensor to be padded.
        target_length (int): The desired length of the output tensor.
    Returns:
        torch.Tensor: The padded audio tensor if padding is needed, otherwise the original tensor.
    Raises:
        Logs an error if the target length is shorter than the audio length.
    """

    if target_length < audio.shape[-1]:
        logging.error("Target length shorter than audio len")
        return audio
    elif target_length == audio.shape[-1]:
        return audio
    else:
        return pad_samples(audio, target_length - audio.shape[-1])


def combine_audio_list(audio: list[torch.Tensor]):
    lens = [x.shape[-1] for x in audio]
    if len(set(lens)) == 1:
        return torch.stack(audio), torch.tensor(lens)
    max_len = max(lens)
    new_audio = []
    for x in audio:
        new_audio.append(pad_tolength(x, max_len))
    return torch.stack(new_audio), torch.tensor(lens)


class STFTWrapper(torch.nn.Module):
    def __init__(
        self, n_fft=1024, hop_length=256, win_length=None, window=None, device="cpu"
    ):
        super(STFTWrapper, self).__init__()

        self.device = device

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.window = torch.hann_window(self.win_length).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        do_reshape = False
        if x.ndim == 3:
            do_reshape = True
            batch, chan, samp = x.shape
            x = x.reshape(batch * chan, samp)

        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        X = torch.view_as_real(X)

        if do_reshape:
            _, F, T, comp = X.shape
            X = X.reshape(batch, chan, F, T, comp)

        return X

    def inverse(
        self, X: Union[torch.Tensor, torch.ComplexFloatStorage]
    ) -> torch.Tensor:
        X = X.contiguous()
        if not X.is_complex():
            X = torch.view_as_complex(X)

        do_reshape = X.ndim == 4
        if do_reshape:
            # Multichannel input
            batch, chan, freq, frames = X.shape
            X = X.reshape(batch * chan, freq, frames)

        x = torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
        )

        if do_reshape:
            x = x.reshape(batch, chan, -1)

        return x
