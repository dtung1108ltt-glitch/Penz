"""Non-blocking audio recording stream using SoundDevice with pre-buffering."""
import logging
import queue
import threading
from typing import Callable, Optional
import numpy as np

from config.audio_config import AudioConfig
from core.ring_buffer import RingBuffer

logger = logging.getLogger(__name__)


class AudioRecorder:
    """Non-blocking microphone stream recorder.
    
    Streams raw audio frames from hardware into a thread-safe queue and a circular ring buffer.
    Pure DSP - does not depend on any AI dependencies.
    """

    def __init__(self, config: Optional[AudioConfig] = None, pre_buffer_duration_ms: int = 300):
        self.config = config or AudioConfig()
        self.sample_rate = self.config.sample_rate
        self.chunk_size = self.config.chunk_size

        # Ring buffer capacity in samples for pre-buffering speech onset
        pre_buffer_samples = int((pre_buffer_duration_ms / 1000.0) * self.sample_rate)
        self.ring_buffer = RingBuffer(capacity_samples=pre_buffer_samples, dtype=np.float32)

        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._stream = None
        self._is_running = False
        self._callbacks: list[Callable[[np.ndarray], None]] = []
        self._lock = threading.Lock()

    def add_frame_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Register a callback to be invoked on every incoming audio frame."""
        self._callbacks.append(callback)

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """SoundDevice InputStream callback executed on an internal OS audio thread."""
        if status:
            logger.warning("Audio input stream status warning: %s", status)

        # Ensure single mono channel, normalized float32
        chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.flatten().copy()
        chunk = chunk.astype(np.float32)

        # Maintain pre-buffer
        self.ring_buffer.write(chunk)

        # Enqueue for downstream consumers
        try:
            self._queue.put_nowait(chunk)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(chunk)
            except Exception:
                pass

        # Trigger registered frame callbacks
        for cb in self._callbacks:
            try:
                cb(chunk)
            except Exception as e:
                logger.error("Error in audio frame callback: %s", e)

    def start(self) -> None:
        """Start capturing audio non-blocking."""
        with self._lock:
            if self._is_running:
                return

            try:
                import sounddevice as sd
                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.chunk_size,
                    channels=self.config.channels,
                    dtype=self.config.dtype,
                    device=self.config.input_device,
                    callback=self._audio_callback,
                )
                self._stream.start()
                self._is_running = True
                logger.info("AudioRecorder started (Sample rate: %d Hz, Chunk: %d).", self.sample_rate, self.chunk_size)
            except (ImportError, Exception) as e:
                logger.warning("Audio input stream unavailable (%s). Running in virtual/dummy mode.", e)
                self._stream = None
                self._is_running = True
                return

    def stop(self) -> None:
        """Stop capturing audio and release hardware resources."""
        with self._lock:
            if not self._is_running:
                return

            self._is_running = False
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.warning("Error stopping audio input stream: %s", e)
                finally:
                    self._stream = None

            logger.info("AudioRecorder stopped.")

    def read_frame(self, timeout: Optional[float] = 0.1) -> Optional[np.ndarray]:
        """Read the next audio frame from the queue."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_pre_buffer(self) -> np.ndarray:
        """Retrieve all currently accumulated pre-buffer samples."""
        return self.ring_buffer.read_all()

    def clear_queue(self) -> None:
        """Clear queued frames."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    @property
    def is_running(self) -> bool:
        return self._is_running
