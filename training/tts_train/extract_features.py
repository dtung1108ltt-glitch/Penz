"""Feature extraction for acoustic modeling & neural vocoders: Mel-spectrogram & F0 pitch."""
import sys
import wave
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def compute_f0_pitch_autocorr(audio: np.ndarray, sample_rate: int = 16000, frame_size: int = 512, hop_size: int = 160) -> np.ndarray:
    """Extract fundamental frequency (F0 pitch) using short-time autocorrelation."""
    min_lag = int(sample_rate / 400)  # Max 400 Hz
    max_lag = int(sample_rate / 60)   # Min 60 Hz

    num_frames = (len(audio) - frame_size) // hop_size + 1
    f0 = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_size
        frame = audio[start : start + frame_size]
        if np.std(frame) < 1e-4:
            continue

        corr = np.correlate(frame, frame, mode="full")[frame_size - 1 :]
        valid_corr = corr[min_lag:max_lag]
        if len(valid_corr) > 0 and np.max(valid_corr) > 0.3 * corr[0]:
            peak_lag = min_lag + np.argmax(valid_corr)
            f0[i] = sample_rate / peak_lag

    return f0


def compute_log_mel(audio: np.ndarray, sample_rate: int = 16000, n_fft: int = 512, hop_size: int = 160, n_mels: int = 80) -> np.ndarray:
    """Extract standard Log-Mel Spectrogram features using scipy or numpy STFT."""
    try:
        from scipy.signal import spectrogram
        _, _, sxx = spectrogram(audio, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_size)
        log_mel = np.log(np.maximum(sxx, 1e-5))
        return log_mel.astype(np.float32)
    except ImportError:
        # NumPy STFT fallback
        num_frames = (len(audio) - n_fft) // hop_size + 1
        window = np.hanning(n_fft)
        spec = []
        for i in range(num_frames):
            frame = audio[i * hop_size : i * hop_size + n_fft] * window
            fft_mag = np.abs(np.fft.rfft(frame))
            spec.append(fft_mag)
        sxx = np.array(spec).T if spec else np.zeros((n_fft // 2 + 1, 1), dtype=np.float32)
        log_mel = np.log(np.maximum(sxx, 1e-5))
        return log_mel.astype(np.float32)


def read_wav(file_path: Path) -> tuple[int, np.ndarray]:
    try:
        import scipy.io.wavfile as wavfile
        sr, raw = wavfile.read(str(file_path))
        return sr, raw
    except ImportError:
        with wave.open(str(file_path), "rb") as wf:
            sr = wf.getframerate()
            raw_bytes = wf.readframes(wf.getnframes())
            dtype = np.int16 if wf.getsampwidth() == 2 else np.float32
            raw = np.frombuffer(raw_bytes, dtype=dtype)
            if wf.getnchannels() > 1:
                raw = raw.reshape(-1, wf.getnchannels())
            return sr, raw


def process_features(input_wav: Path, output_dir: Path):
    """Extract and persist features for a single WAV file."""
    sr, raw = read_wav(input_wav)
    if raw.dtype == np.int16:
        audio = raw.astype(np.float32) / 32768.0
    else:
        audio = raw.astype(np.float32)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    mel = compute_log_mel(audio, sample_rate=sr)
    f0 = compute_f0_pitch_autocorr(audio, sample_rate=sr)

    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = input_wav.stem
    np.save(str(output_dir / f"{base_name}_mel.npy"), mel)
    np.save(str(output_dir / f"{base_name}_f0.npy"), f0)
    print(f"Extracted {base_name}: Mel {mel.shape}, F0 {f0.shape}")


if __name__ == "__main__":
    processed_dir = Path("data/processed")
    features_dir = Path("data/processed/features")
    wav_files = list(processed_dir.glob("*.wav"))

    if not wav_files:
        print("No WAV files found in data/processed/. Generating dummy test file.")
        dummy_wav = processed_dir / "test_synth.wav"
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        dummy_data = (0.2 * np.sin(2 * np.pi * 220 * t) * 32767).astype(np.int16)
        with wave.open(str(dummy_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(dummy_data.tobytes())
        wav_files = [dummy_wav]

    for wav in wav_files:
        process_features(wav, features_dir)
