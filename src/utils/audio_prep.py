import torch
import torchaudio
import logging

from utils.signal_utils import rms_normalize


class AudioPrep:
    def __init__(
        self,
        output_channels: int,
        input_sr: int,
        output_sr: int,
        output_rms: float,
        device: str,
    ):
        """
        Initialize the AudioPrep class.

        Args:
            output_channels (int): Number of output audio channels.
            input_sr (int): Sample rate of the input audio.
            output_sr (int): Desired sample rate for the output audio.
            output_rms (float): Target RMS value for output normalization.
            device (str): Device to run the resampler on (e.g., 'cpu' or 'cuda').

        Attributes:
            output_channels (int): Number of output channels.
            input_sr (int): Input sample rate.
            output_sr (int): Output sample rate.
            output_rms (float): Target output RMS value.
            resampler (torchaudio.transforms.Resample): Resampler for converting audio from input_sr to output_sr.
        """
        self.output_channels = output_channels
        self.input_sr = input_sr
        self.output_sr = output_sr
        self.output_rms = output_rms
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=input_sr, new_freq=output_sr
        ).to(device)

    def process(self, audio: torch.Tensor, fs: int):
        """
        Processes an audio tensor by ensuring correct shape, resampling, and RMS normalization.
        Args:
            audio (torch.Tensor): Input audio tensor. shape: (batch x channels x samples)
            fs (int): Sample rate of the input audio.
        Returns:
            torch.Tensor: Processed audio tensor with correct shape, sample rate, and RMS normalization. shape: (batch (x channels) x samples)
        Behavior:
            - If the input audio is 1D, it is unsqueezed to add a channel dimension.
            - If the input sample rate matches `self.input_sr`, the audio is resampled.
            - If the input sample rate does not match `self.output_sr`, an error is logged.
            - If `self.output_rms` is non-zero, the audio is RMS-normalized to this value.
        """

        if audio.ndim == 1:
            # Assume shape = (samples)
            audio = audio.unsqueeze(0)
        elif audio.ndim > 2:
            logging.error(f"Too many dimensions in audio!!\naudio.shape={audio.shape}")

        if self.output_channels > audio.shape[0]:
            logging.error(
                f"Invalid # of channels. Requested {self.output_channels} channels but audio only has {audio.shape[0]}"
            )
        elif self.output_channels == 1:
            audio = audio[0, :]
        else:
            audio = audio[: self.output_channels, :]

        if fs == self.input_sr:
            audio = self.resampler(audio)
        elif fs != self.output_sr:
            logging.error(
                f"Unexpected sample rate:\nExpected input: {self.input_sr}Hz\nExpected output: {self.output_sr}Hz\nGiven input: {fs}Hz"
            )

        if self.output_rms != 0:
            audio = rms_normalize(audio, self.output_rms)

        return audio
