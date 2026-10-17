"""PyTorch Custom Datasets and Collate Functions for Audio and Text Fine-Tuning."""
import json
from pathlib import Path
from typing import Optional, Union
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object


class AudioSpeechDataset(Dataset):
    """PyTorch Dataset for Speech-to-Text & TTS training.
    
    Loads audio files, resamples to 16kHz, computes log Mel-spectrograms,
    and returns paired text targets.
    """

    def __init__(
        self,
        metadata_jsonl: Union[str, Path],
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for AudioSpeechDataset. Run: pip install torch torchaudio")

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.samples = []

        path = Path(metadata_jsonl)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Union[torch.Tensor, str]]:
        import torchaudio

        item = self.samples[idx]
        audio_path = item["audio_filepath"]
        text = item["text"]

        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        mel_spec = mel_transform(waveform)
        log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

        return {
            "waveform": waveform.squeeze(0),
            "mel_spectrogram": log_mel_spec.squeeze(0),
            "text": text,
        }


class ConversationalDataset(Dataset):
    """PyTorch Dataset for instruction/dialogue fine-tuning of LLMs."""

    def __init__(self, data_path: Union[str, Path], tokenizer=None, max_length: int = 512):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ConversationalDataset.")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversations = []

        path = Path(data_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                if path.suffix == ".jsonl":
                    for line in f:
                        if line.strip():
                            self.conversations.append(json.loads(line))
                elif path.suffix == ".json":
                    self.conversations = json.load(f)

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.conversations[idx]
        prompt = item.get("prompt", item.get("instruction", ""))
        response = item.get("response", item.get("output", ""))
        full_text = f"<s>[INST] {prompt} [/INST] {response} </s>"

        if self.tokenizer is None:
            return {"text": full_text}

        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": encoded["input_ids"].squeeze(0),
        }
