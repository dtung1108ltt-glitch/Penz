"""Non-blocking audio playback stream with immediate interruption / barge-in support."""
import logging
import queue
import threading
from typing import Optional
import numpy as np

from config.audio_config import AudioConfig

logger = logging.getLogger(__name__)


class AudioPlayer:
    """Non-blocking audio output player supporting seamless streaming and instant interruption.
    
    Pure DSP - does not depend on any AI dependencies.
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        self.config = config or AudioConfig()
        self.sample_rate = self.config.sample_rate
        self.chunk_size = self.config.chunk_size

        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self._current_chunk: Optional[np.ndarray] = None
        self._chunk_offset: int = 0

        self._stream = None
        self._is_running = False
        self._is_playing = False
        self._lock = threading.Lock()

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """SoundDevice OutputStream callback running on internal OS audio thread."""
        if status:
            logger.warning("Audio output stream status warning: %s", status)

        needed = frames
        outdata.fill(0)
        out_idx = 0

        while needed > 0 and self._is_running:
            if self._current_chunk is None or self._chunk_offset >= len(self._current_chunk):
                try:
                    # Check if there is next audio chunk available
                    self._current_chunk = self._queue.get_nowait()
                    self._chunk_offset = 0

                    if self._current_chunk is None:
                        # Sentinel received: End of stream
                        self._is_playing = False
                        break
                    self._is_playing = True
                except queue.Empty:
                    self._current_chunk = None
                    self._is_playing = False
                    break

            available = len(self._current_chunk) - self._chunk_offset
            to_copy = min(needed, available)

            chunk_slice = self._current_chunk[self._chunk_offset:self._chunk_offset + to_copy]
            if outdata.ndim > 1:
                outdata[out_idx:out_idx + to_copy, 0] = chunk_slice
            else:
                outdata[out_idx:out_idx + to_copy] = chunk_slice

            self._chunk_offset += to_copy
            out_idx += to_copy
            needed -= to_copy

    def start(self) -> None:
        """Start the audio output stream."""
        with self._lock:
            if self._is_running:
                return

            try:
                import sounddevice as sd
                self._stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    blocksize=self.chunk_size,
                    channels=self.config.channels,
                    dtype=self.config.dtype,
                    device=self.config.output_device,
                    callback=self._audio_callback,
                )
                self._stream.start()
                self._is_running = True
                logger.info("AudioPlayer started (Sample rate: %d Hz).", self.sample_rate)
            except (ImportError, Exception) as e:
                logger.warning("Audio output stream unavailable (%s). Running in virtual/dummy mode.", e)
                self._stream = None
                self._is_running = True
                return

    def play_chunk(self, audio_data: np.ndarray) -> None:
        """Enqueue an audio frame/chunk for playback."""
        if not self._is_running:
            self.start()

        flat_data = audio_data.flatten().astype(np.float32)
        if flat_data.size > 0:
            self._queue.put(flat_data)
            self._is_playing = True

    def mark_stream_end(self) -> None:
        """Put end-of-speech sentinel into the queue."""
        self._queue.put(None)

    def interrupt(self) -> None:
        """Instant Barge-In: immediately abort playback and purge queued audio in <10ms."""
        with self._lock:
            # Drain queue immediately
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break

            self._current_chunk = None
            self._chunk_offset = 0
            self._is_playing = False
            logger.debug("AudioPlayer interrupted (playback buffer purged).")

    def stop(self) -> None:
        """Stop player stream completely."""
        with self._lock:
            self.interrupt()
            self._is_running = False
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.warning("Error closing audio output stream: %s", e)
                finally:
                    self._stream = None
            logger.info("AudioPlayer stopped.")

    @property
    def is_playing(self) -> bool:
        """Return True if audio is actively playing or queued."""
        return self._is_playing or not self._queue.empty()

    @property
    def is_running(self) -> bool:
        return self._is_running
