"""Voice Activity Detection (VAD) using DSP Energy RMS, ZCR, and optional Silero VAD."""
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np

from config.vad_config import VADConfig

logger = logging.getLogger(__name__)


@dataclass
class VADResult:
    """Result of processing a single audio frame."""
    is_speech: bool
    speech_start: bool
    speech_end: bool
    rms_db: float
    zcr: float


class VADDetector:
    """DSP-first Voice Activity Detector with state-machine for speech segmentation.
    
    Operates strictly on raw numpy audio arrays.
    """

    def __init__(self, config: Optional[VADConfig] = None, sample_rate: int = 16000, chunk_size: int = 512):
        self.config = config or VADConfig()
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

        # Duration of one chunk in ms
        self.chunk_duration_ms = (self.chunk_size / self.sample_rate) * 1000.0

        # Frame counts needed to trigger speech start / end
        self.min_speech_frames = max(1, int(self.config.min_speech_duration_ms / self.chunk_duration_ms))
        self.min_silence_frames = max(1, int(self.config.min_silence_duration_ms / self.chunk_duration_ms))

        # State tracking
        self.is_speaking = False
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0

        # Optional Silero VAD model handle
        self._silero_model = None
        if self.config.backend == "silero":
            self._init_silero()

    def _init_silero(self) -> None:
        """Attempt to load lightweight Silero VAD; fallback cleanly if unavailable."""
        try:
            import torch
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=True,
            )
            self._silero_model = model
            logger.info("Silero VAD initialized successfully.")
        except Exception as e:
            logger.warning("Silero VAD initialization failed (%s). Falling back to Pure DSP RMS+ZCR.", e)
            self._silero_model = None

    @staticmethod
    def calculate_rms(audio_chunk: np.ndarray) -> float:
        """Calculate Root Mean Square (RMS) energy of the frame."""
        if audio_chunk.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(audio_chunk, dtype=np.float64))))

    @classmethod
    def calculate_rms_db(cls, audio_chunk: np.ndarray) -> float:
        """Calculate RMS in Decibels (dBFS)."""
        rms = cls.calculate_rms(audio_chunk)
        if rms < 1e-7:
            return -100.0
        return float(20.0 * np.log10(rms))

    @staticmethod
    def calculate_zcr(audio_chunk: np.ndarray) -> float:
        """Calculate Zero Crossing Rate (ZCR)."""
        if audio_chunk.size < 2:
            return 0.0
        zero_crossings = np.nonzero(np.diff(audio_chunk > 0))[0]
        return float(len(zero_crossings) / (audio_chunk.size - 1))

    def _is_frame_speech(self, audio_chunk: np.ndarray) -> tuple[bool, float, float]:
        """Determine if a frame contains speech using DSP metrics or Silero VAD."""
        rms_db = self.calculate_rms_db(audio_chunk)
        zcr = self.calculate_zcr(audio_chunk)

        if self._silero_model is not None:
            try:
                import torch
                tensor_chunk = torch.from_numpy(audio_chunk.astype(np.float32))
                prob = self._silero_model(tensor_chunk, self.sample_rate).item()
                is_active = prob > 0.5
                return is_active, rms_db, zcr
            except Exception:
                pass

        # Pure DSP decision based on energy threshold & ZCR
        is_active = rms_db > self.config.silence_threshold_db and zcr >= self.config.zcr_threshold
        return is_active, rms_db, zcr

    def process_frame(self, audio_chunk: np.ndarray) -> VADResult:
        """Feed one audio chunk into the state machine and detect speech boundaries."""
        is_active, rms_db, zcr = self._is_frame_speech(audio_chunk)

        speech_start = False
        speech_end = False

        if is_active:
            self.consecutive_speech_frames += 1
            self.consecutive_silence_frames = 0

            if not self.is_speaking and self.consecutive_speech_frames >= self.min_speech_frames:
                self.is_speaking = True
                speech_start = True
        else:
            self.consecutive_silence_frames += 1
            if self.consecutive_speech_frames > 0:
                self.consecutive_speech_frames = max(0, self.consecutive_speech_frames - 1)

            if self.is_speaking and self.consecutive_silence_frames >= self.min_silence_frames:
                self.is_speaking = False
                speech_end = True
                self.consecutive_speech_frames = 0
                self.consecutive_silence_frames = 0

        return VADResult(
            is_speech=self.is_speaking,
            speech_start=speech_start,
            speech_end=speech_end,
            rms_db=rms_db,
            zcr=zcr,
        )

    def reset(self) -> None:
        """Reset internal VAD state machine."""
        self.is_speaking = False
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0
