"""Text-To-Speech (TTS) inference engines."""
import asyncio
import io
import logging
from typing import AsyncGenerator, Optional
import numpy as np

from config.model_config import TTSConfig
from engines.base_engine import BaseTTSEngine

logger = logging.getLogger(__name__)


class EdgeTTSEngine(BaseTTSEngine):
    """Text-To-Speech engine using Microsoft Edge-TTS service."""

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self._is_ready = False

    def initialize(self) -> None:
        try:
            import edge_tts  # noqa: F401
            self._is_ready = True
            logger.info("EdgeTTSEngine initialized with voice: %s", self.config.voice)
        except ImportError:
            logger.error("edge-tts package is not installed. Run: pip install edge-tts")
            raise

    def shutdown(self) -> None:
        self._is_ready = False
        logger.info("EdgeTTSEngine shutdown.")

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    async def _text_to_audio_bytes(self, text: str) -> bytes:
        """Fetch MP3 bytes from Edge TTS asynchronously."""
        import edge_tts
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.config.voice,
            rate=self.config.rate,
            pitch=self.config.pitch,
        )
        audio_stream = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_stream.write(chunk["data"])
        return audio_stream.getvalue()

    def _decode_mp3_to_float32(self, mp3_bytes: bytes, target_sample_rate: int = 16000) -> np.ndarray:
        """Decode MP3 audio bytes into 16kHz mono float32 NumPy array."""
        if not mp3_bytes:
            return np.empty(0, dtype=np.float32)

        try:
            # Attempt decoding using soundfile
            import soundfile as sf
            data, sr = sf.read(io.BytesIO(mp3_bytes), dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != target_sample_rate:
                from scipy import signal
                num_samples = int(len(data) * target_sample_rate / sr)
                data = signal.resample(data, num_samples).astype(np.float32)
            return data
        except Exception:
            # Fallback to scipy or pydub if soundfile does not support mp3
            try:
                from pydub import AudioSegment
                audio_seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
                audio_seg = audio_seg.set_frame_rate(target_sample_rate).set_channels(1)
                samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float32)
                # Normalize 16-bit PCM to [-1.0, 1.0]
                return samples / 32768.0
            except Exception as e:
                logger.error("Failed to decode MP3 stream: %s. Returning empty audio.", e)
                return np.empty(0, dtype=np.float32)

    def synthesize(self, text: str) -> np.ndarray:
        """Synchronously synthesize text to float32 audio."""
        if not text.strip():
            return np.empty(0, dtype=np.float32)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                mp3_bytes = pool.submit(lambda: asyncio.run(self._text_to_audio_bytes(text))).result()
        else:
            mp3_bytes = asyncio.run(self._text_to_audio_bytes(text))

        return self._decode_mp3_to_float32(mp3_bytes)

    def synthesize_to_file(self, text: str, output_path: str = "output_response.mp3") -> str:
        """Synthesize text to an audio file (e.g., MP3 or WAV)."""
        from pathlib import Path
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        async def _save():
            import edge_tts
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.config.voice,
                rate=self.config.rate,
                pitch=self.config.pitch,
            )
            await communicate.save(str(out_file))

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                pool.submit(lambda: asyncio.run(_save())).result()
        else:
            asyncio.run(_save())

        logger.info("EdgeTTS synthesized audio saved to: %s", out_file)
        return str(out_file)

    async def synthesize_stream(self, text: str) -> AsyncGenerator[np.ndarray, None]:
        """Stream synthesized audio chunks."""
        if not text.strip():
            return

        mp3_bytes = await self._text_to_audio_bytes(text)
        audio_array = self._decode_mp3_to_float32(mp3_bytes)
        if audio_array.size > 0:
            yield audio_array


class PiperTTSEngine(BaseTTSEngine):
    """Local offline neural TTS engine using Piper ONNX."""

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self._is_ready = False

    def initialize(self) -> None:
        logger.info("Initializing Piper TTS (Local Neural)...")
        self._is_ready = True

    def shutdown(self) -> None:
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def synthesize(self, text: str) -> np.ndarray:
        # Placeholder for local piper binary or python-piper binding
        return np.zeros(16000, dtype=np.float32)

    async def synthesize_stream(self, text: str) -> AsyncGenerator[np.ndarray, None]:
        yield self.synthesize(text)


class MockTTSEngine(BaseTTSEngine):
    """Mock TTS Engine producing a gentle audio tone for testing without network."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._is_ready = False

    def initialize(self) -> None:
        self._is_ready = True
        logger.info("MockTTSEngine initialized.")

    def shutdown(self) -> None:
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def synthesize(self, text: str) -> np.ndarray:
        # Generate 0.3s of a gentle 440Hz sine wave tone
        duration_s = 0.3
        t = np.linspace(0, duration_s, int(self.sample_rate * duration_s), endpoint=False)
        sine_wave = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return sine_wave

    def synthesize_to_file(self, text: str, output_path: str = "output_response.wav") -> str:
        """Synthesize mock tone to audio file."""
        import soundfile as sf
        from pathlib import Path
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        audio = self.synthesize(text)
        sf.write(str(out_file), audio, self.sample_rate)
        logger.info("Mock TTS audio saved to: %s", out_file)
        return str(out_file)

    async def synthesize_stream(self, text: str) -> AsyncGenerator[np.ndarray, None]:
        audio = self.synthesize(text)
        chunk_size = 1024
        for i in range(0, len(audio), chunk_size):
            await asyncio.sleep(0.01)
            yield audio[i : i + chunk_size]


def _patch_torchaudio():
    """Ensure torchaudio loads and saves audio via soundfile without requiring torchcodec."""
    try:
        import soundfile as sf
        import torch
        import torchaudio

        def _sf_load(uri, *args, **kwargs):
            data, sr = sf.read(uri, dtype="float32")
            tensor = torch.from_numpy(data)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            elif tensor.ndim == 2:
                tensor = tensor.t()
            return tensor, sr

        def _sf_save(uri, src, sample_rate, *args, **kwargs):
            if isinstance(src, torch.Tensor):
                src = src.detach().cpu().numpy()
            if src.ndim == 2:
                src = src.T
            sf.write(uri, src, sample_rate)

        torchaudio.load = _sf_load
        torchaudio.save = _sf_save
    except Exception as e:
        logger.debug("Failed to patch torchaudio: %s", e)


class CustomVoiceCloneEngine(BaseTTSEngine):
    """Engine tổng hợp giọng nói nhân bản thật (True Neural Zero-Shot Voice Cloning) bằng F5-TTS Vietnamese.

    Sử dụng kiến trúc DiT (Diffusion Transformer) và Vocos 24kHz của F5-TTS, được tinh chỉnh trên dữ liệu tiếng Việt.
    Mẫu giọng được trích xuất trực tiếp từ reference_24k.wav cùng ngữ cảnh văn bản tương ứng.
    Tuyệt đối KHÔNG sử dụng edge-tts hoặc DSP Pitch/EQ giả lập.
    """

    def __init__(
        self,
        profile_path: str = "models/voice_profile/speaker_profile.json",
        reference_wav: str = "models/voice_profile/reference_24k.wav",
        model_path: str = "models/f5_tts_vi/model_latest.safetensors",
        vocab_path: str = "models/f5_tts_vi/vocab.txt",
        sample_rate: int = 24000,
        device: Optional[str] = None,
        nfe_step: int = 32,
        ode_method: str = "euler",
        speed: float = 1.0,
        voice_base: Optional[str] = None,
        use_f5_tts: bool = True,
        **kwargs,
    ):
        from pathlib import Path
        p = Path(profile_path)
        ref = Path(reference_wav)

        if p.suffix.lower() in (".wav", ".mp3", ".flac"):
            self.reference_wav = p
            json_candidate = p.parent / "speaker_profile.json"
            self.profile_path = json_candidate if json_candidate.exists() else p
        elif p.is_dir():
            self.profile_path = p / "speaker_profile.json"
            self.reference_wav = p / "reference_24k.wav"
        else:
            self.profile_path = p
            if ref.suffix.lower() in (".wav", ".mp3", ".flac"):
                self.reference_wav = ref
            else:
                self.reference_wav = self.profile_path.parent / "reference_24k.wav"

        # Check model and vocab paths (relative to cwd or parent)
        m_path = Path(model_path)
        if not m_path.exists():
            cand = self.profile_path.parent.parent / "f5_tts_vi" / "model_latest.safetensors"
            if cand.exists():
                m_path = cand
        self.model_path = m_path

        v_path = Path(vocab_path)
        if not v_path.exists():
            cand = self.profile_path.parent.parent / "f5_tts_vi" / "vocab.txt"
            if cand.exists():
                v_path = cand
        self.vocab_path = v_path

        self.sample_rate = sample_rate
        self.device = device
        self.nfe_step = nfe_step
        self.ode_method = ode_method
        self.speed = speed
        self.profile: dict = {}
        self.ref_text: str = ""
        self.voice_base = "F5-TTS Vietnamese (True Zero-Shot)"
        self.pitch_str = "0Hz (Neural Pitch Contour)"
        self.rate_str = "+0%"
        self.f5tts = None
        self._is_ready = False

    def initialize(self) -> None:
        """Nạp hồ sơ giọng nói và khởi tạo mô hình F5-TTS tiếng Việt."""
        import json
        import torch
        _patch_torchaudio()

        # 1. Nạp hồ sơ âm học từ speaker_profile.json
        if self.profile_path.exists():
            try:
                with open(self.profile_path, "r", encoding="utf-8") as f:
                    self.profile = json.load(f)
                self.ref_text = (
                    self.profile.get("ref_text")
                    or self.profile.get("reference_transcript")
                    or ""
                ).strip()
            except Exception as e:
                logger.warning("Không thể đọc profile %s: %s", self.profile_path, e)

        # Mặc định ref_text nếu chưa có
        if not self.ref_text:
            self.ref_text = (
                "Giữa triệu người lướt qua vô tình ta đã tìm thấy nhau, đã nhiều lần em sai nhưng hôm nay em chắc mình đúng."
            )

        if not self.reference_wav.exists():
            raise FileNotFoundError(
                f"Không tìm thấy reference_wav tại {self.reference_wav} để làm mẫu giọng cho F5-TTS."
            )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint F5-TTS tại {self.model_path}!"
            )

        # 2. Thiết bị tính toán
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            "Khởi tạo F5-TTS Vietnamese (ckpt=%s, vocab=%s, device=%s)...",
            self.model_path,
            self.vocab_path,
            self.device,
        )

        try:
            from f5_tts.api import F5TTS
            self.f5tts = F5TTS(
                model="F5TTS_Base",
                ckpt_file=str(self.model_path),
                vocab_file=str(self.vocab_path),
                ode_method=self.ode_method,
                device=self.device,
            )
            self._is_ready = True
            logger.info("CustomVoiceCloneEngine (F5-TTS True Zero-Shot) đã sẵn sàng.")
        except Exception as e:
            logger.error("Lỗi khi nạp mô hình F5-TTS: %s", e)
            self._is_ready = False
            raise RuntimeError(
                f"Không thể khởi tạo mô hình F5-TTS tiếng Việt: {e}. "
                "Vui lòng kiểm tra checkpoint và thư viện phụ trợ."
            ) from e

    def shutdown(self) -> None:
        self._is_ready = False
        self.f5tts = None
        logger.info("CustomVoiceCloneEngine shutdown.")

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def synthesize(self, text: str, speed: Optional[float] = None, **kwargs) -> np.ndarray:
        """Tổng hợp câu nói bằng mô hình Neural F5-TTS Zero-Shot Voice Cloning.

        Args:
            text: Văn bản tiếng Việt cần tổng hợp.
            speed: Tốc độ đọc (mặc định 1.0).

        Returns:
            np.ndarray: Dữ liệu âm thanh dạng float32 (chuẩn 24000Hz mono).

        Raises:
            RuntimeError: Nếu F5-TTS gặp sự cố (tuyệt đối không âm thầm fallback về edge-tts).
        """
        if not text.strip():
            return np.empty(0, dtype=np.float32)

        if not self._is_ready or self.f5tts is None:
            self.initialize()

        current_speed = speed if speed is not None else self.speed

        try:
            logger.info("F5-TTS bắt đầu tổng hợp: '%s' (speed=%.2f)...", text, current_speed)
            wav, sr, _ = self.f5tts.infer(
                ref_file=str(self.reference_wav),
                ref_text=self.ref_text,
                gen_text=text,
                speed=current_speed,
                nfe_step=self.nfe_step,
                show_info=lambda *a, **k: None,
            )

            if wav.ndim > 1:
                wav = np.mean(wav, axis=1)

            # Đảm bảo sample rate đồng bộ với self.sample_rate
            if sr != self.sample_rate:
                from scipy import signal
                num_samples = int(len(wav) * self.sample_rate / sr)
                wav = signal.resample(wav, num_samples).astype(np.float32)

            return wav.astype(np.float32)
        except Exception as e:
            logger.error("F5-TTS Voice Cloning inference thất bại: %s", e, exc_info=True)
            raise RuntimeError(
                f"F5-TTS Voice Cloning inference thất bại khi tổng hợp '{text}': {e}. "
                "Hệ thống không sử dụng fallback giả lập."
            ) from e

    def synthesize_to_file(self, text: str, output_path: str = "output_response.wav", speed: Optional[float] = None, **kwargs) -> str:
        """Tổng hợp văn bản ra tệp âm thanh WAV bằng F5-TTS."""
        import soundfile as sf
        from pathlib import Path
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        audio = self.synthesize(text, speed=speed)
        sf.write(str(out_file), audio, self.sample_rate)
        logger.info("F5-TTS synthesized audio saved to: %s", out_file)
        return str(out_file)

    async def synthesize_stream(self, text: str) -> AsyncGenerator[np.ndarray, None]:
        audio = self.synthesize(text)
        chunk_size = 2048
        for i in range(0, len(audio), chunk_size):
            await asyncio.sleep(0.01)
            yield audio[i : i + chunk_size]


def get_tts_engine(config: Optional[TTSConfig] = None) -> BaseTTSEngine:
    """Factory function for TTS engines."""
    cfg = config or TTSConfig()
    if cfg.backend == "mock":
        return MockTTSEngine()
    elif cfg.backend == "edge-tts":
        return EdgeTTSEngine(cfg)
    elif cfg.backend == "piper":
        return PiperTTSEngine(cfg)
    elif cfg.backend in ("custom", "voice-clone", "custom-voice", "clone", "f5-tts", "f5_tts"):
        profile_path = getattr(cfg, "voice_profile_path", "models/voice_profile/speaker_profile.json")
        reference_wav = getattr(cfg, "reference_wav_path", "models/voice_profile/reference_24k.wav")
        return CustomVoiceCloneEngine(
            profile_path=str(profile_path),
            reference_wav=str(reference_wav),
            voice_base=getattr(cfg, "voice", None),
        )
    else:
        logger.warning("Unknown TTS backend '%s', defaulting to MockTTSEngine.", cfg.backend)
        return MockTTSEngine()

