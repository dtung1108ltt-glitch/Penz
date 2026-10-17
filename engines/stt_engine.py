"""Speech-To-Text (STT) inference engines."""
import logging
from typing import Optional
import numpy as np

from config.model_config import STTConfig
from engines.base_engine import BaseSTTEngine

logger = logging.getLogger(__name__)


class FasterWhisperSTTEngine(BaseSTTEngine):
    """Production STT Engine powered by CTranslate2 (Faster-Whisper)."""

    def __init__(self, config: Optional[STTConfig] = None):
        self.config = config or STTConfig()
        self.model = None
        self._is_ready = False

    def initialize(self) -> None:
        """Load Faster-Whisper model into memory."""
        try:
            from faster_whisper import WhisperModel
            logger.info(
                "Loading Faster-Whisper model '%s' on device '%s' (%s)...",
                self.config.model_size,
                self.config.device,
                self.config.compute_type,
            )
            self.model = WhisperModel(
                model_size_or_path=self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
            self._is_ready = True
            logger.info("Faster-Whisper model loaded successfully.")
        except ImportError:
            logger.error("faster-whisper is not installed. Run: pip install faster-whisper")
            raise
        except Exception as e:
            logger.error("Failed to initialize Faster-Whisper: %s", e)
            raise

    def shutdown(self) -> None:
        """Release resources."""
        self.model = None
        self._is_ready = False
        logger.info("FasterWhisperSTTEngine shutdown.")

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def transcribe(self, audio_data, sample_rate: int = 16000) -> str:
        """Transcribe audio array or audio file path to text using Faster-Whisper."""
        if not self._is_ready or self.model is None:
            raise RuntimeError("FasterWhisperSTTEngine is not initialized. Call initialize() first.")

        from pathlib import Path
        if isinstance(audio_data, (str, Path)):
            import librosa
            audio_flat, _ = librosa.load(str(audio_data), sr=16000)
        elif isinstance(audio_data, np.ndarray):
            if audio_data.size == 0:
                return ""
            audio_flat = audio_data.astype(np.float32)
            if audio_flat.ndim > 1:
                audio_flat = np.mean(audio_flat, axis=1)
            if sample_rate != 16000 and len(audio_flat) > 0:
                import librosa
                audio_flat = librosa.resample(audio_flat, orig_sr=sample_rate, target_sr=16000)
        else:
            raise ValueError(f"Unsupported audio_data type: {type(audio_data)}")

        if audio_flat.size == 0:
            return ""

        # Normalize audio array to float32 in [-1.0, 1.0]
        max_val = np.max(np.abs(audio_flat))
        if max_val > 1.0:
            audio_flat = audio_flat / max_val

        language = None if self.config.language == "auto" else self.config.language
        segments, info = self.model.transcribe(
            audio_flat,
            language=language,
            beam_size=self.config.beam_size,
            vad_filter=False,  # Core layer already performed VAD segmentation
        )

        text_parts = [segment.text.strip() for segment in segments]
        transcription = " ".join(text_parts).strip()
        logger.debug("Transcribed (%s, prob=%.2f): %s", info.language, info.language_probability, transcription)
        return transcription

    def transcribe_file(self, file_path) -> str:
        """Transcribe an audio file directly from path."""
        return self.transcribe(file_path, sample_rate=16000)


class MockSTTEngine(BaseSTTEngine):
    """Mock STT Engine for testing without downloading neural weights."""

    def __init__(self, mock_response: str = "Xin chào, tôi là trợ lý ảo"):
        self.mock_response = mock_response
        self._is_ready = False

    def initialize(self) -> None:
        self._is_ready = True
        logger.info("MockSTTEngine initialized.")

    def shutdown(self) -> None:
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        return self.mock_response


def get_stt_engine(config: Optional[STTConfig] = None) -> BaseSTTEngine:
    """Factory function for STT engines."""
    cfg = config or STTConfig()
    if cfg.backend == "mock":
        return MockSTTEngine()
    elif cfg.backend == "faster-whisper":
        return FasterWhisperSTTEngine(cfg)
    else:
        logger.warning("Unknown STT backend '%s', defaulting to MockSTTEngine.", cfg.backend)
        return MockSTTEngine()
