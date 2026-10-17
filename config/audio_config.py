"""Audio configuration module."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioConfig:
    """Hardware and signal configuration for audio capture and playback."""
    sample_rate: int = 16000          # 16 kHz: Standard for Whisper, VAD, and speech models
    chunk_size: int = 512             # 512 samples = 32ms latency frame at 16kHz
    channels: int = 1                 # Mono channel
    dtype: str = "float32"            # Internal processing format (normalized -1.0 to 1.0)
    input_device: Optional[int] = None   # None for default OS microphone
    output_device: Optional[int] = None  # None for default OS speaker

    @property
    def chunk_duration_ms(self) -> float:
        """Calculate chunk duration in milliseconds."""
        return (self.chunk_size / self.sample_rate) * 1000.0

    @property
    def bytes_per_sample(self) -> int:
        """Size of one sample in bytes."""
        return 4 if self.dtype == "float32" else 2
