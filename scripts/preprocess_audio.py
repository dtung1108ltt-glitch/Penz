"""Audio preprocessing script: format standardization, silence trimming, and peak normalization."""
import os
import sys
import wave
from pathlib import Path
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.vad_detector import VADDetector


def read_wav(file_path: Path) -> tuple[int, np.ndarray]:
    """Read WAV file using scipy if available, or built-in wave module."""
    try:
        import scipy.io.wavfile as wavfile
        sr, data = wavfile.read(str(file_path))
        return sr, data
    except ImportError:
        with wave.open(str(file_path), "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw_bytes = wf.readframes(n_frames)
            dtype = np.int16 if sampwidth == 2 else (np.uint8 if sampwidth == 1 else np.int32)
            data = np.frombuffer(raw_bytes, dtype=dtype)
            if n_channels > 1:
                data = data.reshape(-1, n_channels)
            return sr, data


def write_wav(file_path: Path, sample_rate: int, data_int16: np.ndarray) -> None:
    """Save 16-bit PCM WAV file."""
    try:
        import scipy.io.wavfile as wavfile
        wavfile.write(str(file_path), sample_rate, data_int16)
    except ImportError:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(file_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(data_int16.tobytes())


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample 1D audio array to target sample rate using scipy or pure numpy interpolation."""
    if orig_sr == target_sr:
        return audio
    num_samples = int(len(audio) * target_sr / orig_sr)
    try:
        from scipy import signal
        return signal.resample(audio, num_samples).astype(np.float32)
    except ImportError:
        x_orig = np.linspace(0, 1, len(audio), endpoint=False)
        x_target = np.linspace(0, 1, num_samples, endpoint=False)
        return np.interp(x_target, x_orig, audio).astype(np.float32)


def normalize_peak(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize peak audio amplitude to avoid clipping."""
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        return (audio / peak) * target_peak
    return audio


def trim_silence(audio: np.ndarray, sample_rate: int = 16000, threshold_db: float = -40.0) -> np.ndarray:
    """Trim leading and trailing silence based on energy RMS in dB."""
    chunk_size = 512
    n_chunks = len(audio) // chunk_size
    if n_chunks == 0:
        return audio

    active_indices = []
    for i in range(n_chunks):
        chunk = audio[i * chunk_size : (i + 1) * chunk_size]
        rms_db = VADDetector.calculate_rms_db(chunk)
        if rms_db > threshold_db:
            active_indices.append(i)

    if not active_indices:
        return audio

    start_idx = max(0, (active_indices[0] - 2) * chunk_size)
    end_idx = min(len(audio), (active_indices[-1] + 3) * chunk_size)
    return audio[start_idx:end_idx]


def preprocess_file(input_path: Path, output_path: Path, target_sr: int = 16000) -> bool:
    """Read, convert to mono, resample, trim, normalize and save."""
    try:
        sr, raw = read_wav(input_path)
        # Convert to float32 [-1.0, 1.0]
        if raw.dtype == np.int16:
            audio = raw.astype(np.float32) / 32768.0
        elif raw.dtype == np.int32:
            audio = raw.astype(np.float32) / 2147483648.0
        else:
            audio = raw.astype(np.float32)

        # Convert stereo to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample
        audio = resample_audio(audio, sr, target_sr)

        # Trim silence
        audio = trim_silence(audio, sample_rate=target_sr)

        # Normalize
        audio = normalize_peak(audio)

        # Save as 16-bit PCM WAV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        int16_audio = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        write_wav(output_path, target_sr, int16_audio)
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def main():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    audio_files = list(raw_dir.glob("*.wav")) + list(raw_dir.glob("*.flac"))
    print(f"🔄 Tìm thấy {len(audio_files)} tệp âm thanh trong {raw_dir}")

    success_count = 0
    for file_path in audio_files:
        dest_path = processed_dir / file_path.name
        if preprocess_file(file_path, dest_path):
            success_count += 1

    print(f"✅ Hoàn tất xử lý {success_count}/{len(audio_files)} tệp âm thanh -> {processed_dir}")


if __name__ == "__main__":
    main()
