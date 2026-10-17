"""Model configuration module for STT, LLM, and TTS inference."""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class STTConfig:
    """Configuration for Speech-to-Text inference."""
    backend: str = os.getenv("STT_BACKEND", "faster-whisper")   # 'faster-whisper', 'whisper', 'mock'
    model_size: str = os.getenv("STT_MODEL_SIZE", "base")       # 'tiny', 'base', 'small', 'medium', 'large-v3'
    device: str = os.getenv("DEVICE", "cpu")                   # 'cuda', 'cpu', 'auto'
    compute_type: str = os.getenv("COMPUTE_TYPE", "int8")      # 'float16', 'int8', 'float32'
    language: str = os.getenv("STT_LANGUAGE", "vi")            # 'vi', 'en', 'auto'
    beam_size: int = 2


@dataclass
class LLMConfig:
    """Configuration for Large Language Model inference."""
    provider: str = os.getenv("LLM_PROVIDER", "local")          # 'local', 'ollama', 'openai', 'mock'
    model_name: str = os.getenv("OLLAMA_MODEL", "llama3:8b-instruct-q4_K_M")
    host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    api_key: Optional[str] = os.getenv("OPENAI_API_KEY", None)
    local_model_path: str = os.getenv("LOCAL_MODEL_PATH", "models/penz_voice_llm_merged")
    temperature: float = 0.6
    max_tokens: int = 256
    system_prompt: str = (
        "Bạn là trợ lý ảo giọng nói tiếng Việt thông minh, phản hồi ngắn gọn, "
        "tự nhiên và súc tích trong 1-2 câu để người nghe dễ tiếp nhận qua giọng nói."
    )


@dataclass
class TTSConfig:
    """Configuration for Text-to-Speech synthesis."""
    backend: str = os.getenv("TTS_BACKEND", "custom")          # 'custom', 'edge-tts', 'piper', 'mock'
    voice: Optional[str] = os.getenv("TTS_VOICE", None)        # Base voice or None for profile-driven selection
    rate: str = os.getenv("TTS_RATE", "+0%")
    pitch: str = os.getenv("TTS_PITCH", "+0Hz")
    piper_model_path: Optional[str] = os.getenv("PIPER_MODEL_PATH", None)
    voice_profile_path: str = os.getenv("VOICE_PROFILE_PATH", "models/voice_profile/speaker_profile.json")
    reference_wav_path: str = os.getenv("REFERENCE_WAV_PATH", "models/voice_profile/reference_24k.wav")


@dataclass
class ModelConfig:
    """Root configuration holding STT, LLM, and TTS settings."""
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
