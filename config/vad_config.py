"""Voice Activity Detection (VAD) configuration."""
import os
from dataclasses import dataclass


@dataclass
class VADConfig:
    """Parameters for Voice Activity Detection and speech boundary segmentation."""
    backend: str = os.getenv("VAD_BACKEND", "energy")          # 'energy', 'silero'
    silence_threshold_db: float = -42.0                       # RMS dB threshold below which audio is silence
    zcr_threshold: float = 0.08                               # Zero crossing rate threshold
    min_silence_duration_ms: int = 500                        # Duration of silence to declare user finished speaking
    min_speech_duration_ms: int = 250                         # Minimum duration to count as valid speech
    pre_buffer_duration_ms: int = 300                         # Duration to keep before detected speech onset (consonants)
    max_speech_duration_s: float = 15.0                       # Maximum duration before forced cutoff
