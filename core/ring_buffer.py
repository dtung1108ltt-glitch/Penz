"""Thread-safe circular ring buffer for pre-buffering raw audio frames."""
import threading
from typing import Optional
import numpy as np


class RingBuffer:
    """Thread-safe circular buffer for NumPy audio arrays.
    
    Used to hold a pre-buffer of audio (e.g. 300ms) before VAD speech detection,
    ensuring speech onset consonants (plosives/fricatives) are not clipped.
    """

    def __init__(self, capacity_samples: int, dtype: np.dtype = np.float32):
        """Initialize circular ring buffer with fixed capacity in samples."""
        self.capacity = max(1, capacity_samples)
        self.dtype = dtype
        self.buffer = np.zeros(self.capacity, dtype=dtype)
        self.write_idx = 0
        self.size = 0
        self.lock = threading.Lock()

    def write(self, data: np.ndarray) -> None:
        """Write 1D audio sample array into the ring buffer."""
        if data.size == 0:
            return

        with self.lock:
            flat_data = data.flatten().astype(self.dtype)
            n_samples = flat_data.size

            if n_samples >= self.capacity:
                # If incoming data exceeds capacity, retain only the newest tail
                self.buffer[:] = flat_data[-self.capacity:]
                self.write_idx = 0
                self.size = self.capacity
                return

            end_idx = self.write_idx + n_samples
            if end_idx <= self.capacity:
                self.buffer[self.write_idx:end_idx] = flat_data
            else:
                first_part = self.capacity - self.write_idx
                self.buffer[self.write_idx:] = flat_data[:first_part]
                second_part = n_samples - first_part
                self.buffer[:second_part] = flat_data[first_part:]

            self.write_idx = (self.write_idx + n_samples) % self.capacity
            self.size = min(self.capacity, self.size + n_samples)

    def read_all(self) -> np.ndarray:
        """Read all currently stored data in chronological order without clearing."""
        with self.lock:
            if self.size == 0:
                return np.empty(0, dtype=self.dtype)

            if self.size < self.capacity:
                # Not wrapped yet
                return self.buffer[:self.size].copy()

            # Wrapped around: reconstruct oldest to newest
            return np.concatenate((self.buffer[self.write_idx:], self.buffer[:self.write_idx]))

    def clear(self) -> None:
        """Clear the ring buffer."""
        with self.lock:
            self.write_idx = 0
            self.size = 0
            self.buffer.fill(0)

    def __len__(self) -> int:
        with self.lock:
            return self.size
