"""Abstract Base Classes for inference engines adhering to Dependency Inversion."""
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional
import numpy as np


class BaseEngine(ABC):
    """Root interface for all inference engines."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize engine, allocate hardware memory, and load model weights."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Release GPU/CPU resources, unload model weights, and clean up."""
        pass

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if engine is initialized and ready for inference."""
        pass


class BaseSTTEngine(BaseEngine):
    """Abstract interface for Speech-to-Text inference."""

    @abstractmethod
    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe raw audio waveform into text.
        
        Args:
            audio_data: 1D NumPy array of audio samples (float32).
            sample_rate: Audio sampling rate in Hz (default: 16000).
            
        Returns:
            Transcribed text string.
        """
        pass


class BaseLLMEngine(BaseEngine):
    """Abstract interface for conversational Language Model inference."""

    @abstractmethod
    def generate(self, prompt: str, history: Optional[list[dict[str, str]]] = None) -> str:
        """Generate complete response text synchronously."""
        pass

    @abstractmethod
    async def generate_stream(
        self, prompt: str, history: Optional[list[dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream response chunks (tokens or clauses) asynchronously.
        
        Yields:
            Incremental text tokens or sentence chunks.
        """
        pass


class BaseTTSEngine(BaseEngine):
    """Abstract interface for Text-to-Speech synthesis."""

    @abstractmethod
    def synthesize(self, text: str) -> np.ndarray:
        """Synthesize text into raw float32 audio waveform (16kHz mono)."""
        pass

    @abstractmethod
    async def synthesize_stream(self, text: str) -> AsyncGenerator[np.ndarray, None]:
        """Stream synthesized audio chunks asynchronously as they become available.
        
        Yields:
            1D NumPy array of audio frames (float32, 16kHz).
        """
        pass
