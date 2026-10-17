"""
Module đăng ký và chuẩn hóa hồ sơ giọng nói tùy chỉnh (Voice Enrollment & Profile Extraction).
Hỗ trợ mọi định dạng âm thanh / video (.wav, .mp3, .m4a, .mp4).
Chuẩn hóa về 24kHz Mono PCM và trích xuất F0 Pitch, Energy, Duration vào speaker_profile.json.
"""
import argparse
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import wave
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Terminal UTF-8 trên Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Voice_Enrollment")

DEFAULT_OUTPUT_WAV = PROJECT_ROOT / "models" / "voice_profile" / "reference_24k.wav"
DEFAULT_PROFILE_JSON = PROJECT_ROOT / "models" / "voice_profile" / "speaker_profile.json"


def compute_f0_autocorr(
    audio: np.ndarray,
    sample_rate: int = 24000,
    frame_size_ms: float = 30.0,
    hop_size_ms: float = 10.0,
    f0_min: float = 65.0,
    f0_max: float = 450.0,
) -> np.ndarray:
    """Trích xuất đường bao cao độ F0 (Pitch) bằng kỹ thuật Autocorrelation.
    
    Args:
        audio: Mảng tín hiệu âm thanh 1 chiều (float32 trong khoảng [-1.0, 1.0]).
        sample_rate: Tần số lấy mẫu (Hz).
        frame_size_ms: Kích thước khung tính toán (mili-giây).
        hop_size_ms: Bước nhảy giữa các khung (mili-giây).
        f0_min: Tần số cơ bản tối thiểu (Hz).
        f0_max: Tần số cơ bản tối đa (Hz).
        
    Returns:
        Mảng numpy chứa tần số F0 (Hz) cho từng khung thời gian (0 đối với khung vô thanh).
    """
    frame_size = int(sample_rate * frame_size_ms / 1000.0)
    hop_size = int(sample_rate * hop_size_ms / 1000.0)
    
    min_lag = int(sample_rate / f0_max)
    max_lag = int(sample_rate / f0_min)
    
    if len(audio) < frame_size:
        return np.array([0.0], dtype=np.float32)
        
    num_frames = (len(audio) - frame_size) // hop_size + 1
    f0 = np.zeros(num_frames, dtype=np.float32)
    
    # Cửa sổ Hanning làm mượt biên
    window = np.hanning(frame_size)
    
    for i in range(num_frames):
        start = i * hop_size
        frame = audio[start : start + frame_size] * window
        energy = np.sum(frame ** 2)
        
        # Bỏ qua khung quá yên tĩnh (ngưỡng năng lượng vô thanh)
        if energy < 1e-4:
            continue
            
        corr = np.correlate(frame, frame, mode="full")[frame_size - 1 :]
        if corr[0] <= 1e-8:
            continue
            
        # Chuẩn hóa autocorrelation
        norm_corr = corr / corr[0]
        valid_range = norm_corr[min_lag : min(max_lag, len(norm_corr))]
        
        if len(valid_range) > 0:
            peak_idx = np.argmax(valid_range)
            peak_val = valid_range[peak_idx]
            # Ngưỡng phát hiện tiếng hữu thanh (Voiced sound threshold)
            if peak_val > 0.35:
                lag = min_lag + peak_idx
                f0[i] = sample_rate / lag
                
    return f0


def load_and_convert_to_24k_mono(input_path: Path) -> np.ndarray:
    """Nạp file âm thanh hoặc trích xuất từ video, chuyển đổi về 24kHz Mono float32 [-1.0, 1.0]."""
    if not input_path.exists():
        raise FileNotFoundError(f"Tệp không tồn tại: {input_path}")
        
    suffix = input_path.suffix.lower()
    audio_data = None
    orig_sr = None

    # Thử nạp bằng soundfile trước (nếu là wav/flac/ogg)
    try:
        import soundfile as sf
        raw_data, orig_sr = sf.read(str(input_path), dtype="float32")
        if raw_data.ndim > 1:
            raw_data = np.mean(raw_data, axis=1)
        audio_data = raw_data
    except Exception:
        audio_data = None

    # Nếu soundfile không hỗ trợ (mp3, m4a, mp4, v.v.), dùng pydub
    if audio_data is None:
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(str(input_path))
            seg = seg.set_channels(1)
            orig_sr = seg.frame_rate
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            max_val = float(1 << (8 * seg.sample_width - 1))
            audio_data = samples / max_val
        except Exception as e:
            logger.debug("Pydub không nạp được: %s", e)

    # Nếu pydub không nạp được, thử gọi ffmpeg trực tiếp
    if audio_data is None and shutil.which("ffmpeg"):
        try:
            cmd = [
                "ffmpeg", "-y", "-i", str(input_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1",
                "-f", "wav", "pipe:1"
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            with wave.open(io.BytesIO(res.stdout), "rb") as wf:
                orig_sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                raw_int16 = np.frombuffer(frames, dtype=np.int16)
                audio_data = raw_int16.astype(np.float32) / 32768.0
        except Exception as e:
            logger.debug("FFmpeg subprocess không thành công: %s", e)

    # Thử scipy.io.wavfile nếu là file .wav
    if audio_data is None and suffix == ".wav":
        try:
            import scipy.io.wavfile as wavfile
            orig_sr, raw_int = wavfile.read(str(input_path))
            if raw_int.ndim > 1:
                raw_int = np.mean(raw_int, axis=1)
            if raw_int.dtype == np.int16:
                audio_data = raw_int.astype(np.float32) / 32768.0
            elif raw_int.dtype == np.int32:
                audio_data = raw_int.astype(np.float32) / 2147483648.0
            else:
                audio_data = raw_int.astype(np.float32)
        except Exception as e:
            logger.error("Không thể đọc tệp WAV bằng scipy: %s", e)

    if audio_data is None or len(audio_data) == 0:
        raise ValueError(
            f"Không thể giải mã dữ liệu âm thanh từ tệp: {input_path}. "
            "Vui lòng đảm bảo tệp chứa luồng âm thanh hợp lệ (wav, mp3, m4a, mp4) hoặc cài đặt pydub/soundfile."
        )

    # Resample về đúng 24000 Hz nếu khác
    target_sr = 24000
    if orig_sr != target_sr:
        from scipy import signal
        target_length = int(len(audio_data) * target_sr / orig_sr)
        audio_24k = signal.resample(audio_data, target_length).astype(np.float32)
    else:
        audio_24k = audio_data.astype(np.float32)

    # Chuẩn hóa biên độ đỉnh (Peak Normalization) về khoảng [-0.95, 0.95]
    peak = np.max(np.abs(audio_24k))
    if peak > 1e-4:
        audio_24k = (audio_24k / peak) * 0.95

    return audio_24k


def enroll_voice(
    input_file: str | Path,
    output_wav_path: str | Path = DEFAULT_OUTPUT_WAV,
    profile_json_path: str | Path = DEFAULT_PROFILE_JSON,
    target_sample_rate: int = 24000,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    ref_text: Optional[str] = None,
    target_speaking_f0: Optional[float] = 230.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Đăng ký mẫu giọng nói từ tệp bất kỳ, trích xuất hồ sơ âm học và lưu cấu hình.
    
    Args:
        input_file: Đường dẫn tệp đầu vào (.wav, .mp3, .m4a, .mp4, v.v.).
        output_wav_path: Đường dẫn lưu file reference_24k.wav.
        profile_json_path: Đường dẫn lưu speaker_profile.json.
        target_sample_rate: Tần số lấy mẫu mục tiêu (24000 Hz chuẩn cho Voice Cloning).
        start_sec: Giây bắt đầu cắt mẫu (tùy chọn).
        end_sec: Giây kết thúc cắt mẫu (tùy chọn).
        ref_text: Bản dịch/văn bản chính xác của đoạn âm thanh tham chiếu (tùy chọn).
        target_speaking_f0: Tần số F0 mục tiêu cho giọng nói tự nhiên (mặc định 230Hz cho nữ trẻ).
        
    Returns:
        Tuple gồm (mảng âm thanh float32 24kHz, dictionary hồ sơ người nói).
    """
    in_path = Path(input_file).resolve()
    out_wav = Path(output_wav_path).resolve()
    out_json = Path(profile_json_path).resolve()

    logger.info("=" * 65)
    logger.info("🎙️ BẮT ĐẦU ĐĂNG KÝ HỒ SƠ GIỌNG NÓI TÙY CHỈNH (VOICE ENROLLMENT)")
    logger.info("• Tệp nguồn: %s", in_path)
    if start_sec is not None and end_sec is not None:
        logger.info("• Khoảng thời gian cắt mẫu: %.2fs -> %.2fs (%.2fs)", start_sec, end_sec, end_sec - start_sec)
    
    if not in_path.exists():
        err_msg = (
            f"\n❌ LỖI: Không tìm thấy tệp giọng nói tại:\n   --> {in_path}\n"
            f"Vui lòng sao chép tệp ghi âm giọng nói của bạn (định dạng .wav, .mp3, .m4a hoặc .mp4)\n"
            f"vào thư mục trên hoặc cập nhật biến CUSTOM_VOICE_FILE trong Cell Cấu hình."
        )
        logger.error(err_msg)
        raise FileNotFoundError(err_msg)

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # 1. Chuyển đổi và chuẩn hóa về 24kHz Mono float32
    audio_24k = load_and_convert_to_24k_mono(in_path)

    # Cắt phân đoạn chuẩn nếu có yêu cầu thời gian
    if start_sec is not None and end_sec is not None:
        start_idx = max(0, int(start_sec * target_sample_rate))
        end_idx = min(len(audio_24k), int(end_sec * target_sample_rate))
        audio_24k = audio_24k[start_idx:end_idx]

        # Lọc bỏ khoảng lặng tĩnh / noise ở đầu và cuối
        try:
            import librosa
            audio_24k, _ = librosa.effects.trim(audio_24k, top_db=32, frame_length=512, hop_length=128)
            fade_len = int(target_sample_rate * 0.02)
            if len(audio_24k) > 2 * fade_len:
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = np.linspace(1, 0, fade_len)
                audio_24k[:fade_len] *= fade_in
                audio_24k[-fade_len:] *= fade_out
        except Exception as e:
            logger.debug("Lọc nhiễu librosa: %s", e)

        # Chuẩn hóa biên độ đỉnh
        peak = np.max(np.abs(audio_24k))
        if peak > 1e-4:
            audio_24k = (audio_24k / peak) * 0.95

    duration_s = round(float(len(audio_24k) / target_sample_rate), 2)
    logger.info("Đã chuẩn hóa về 24kHz Mono. Thời lượng: %.2f giây (%d mẫu).", duration_s, len(audio_24k))

    # 2. Lưu tệp WAV 16-bit PCM chất lượng cao
    import scipy.io.wavfile as wavfile
    audio_int16 = (np.clip(audio_24k, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(str(out_wav), target_sample_rate, audio_int16)
    logger.info("Đã lưu tệp âm thanh chuẩn: %s", out_wav)

    # 3. Trích xuất đặc trưng âm học F0 Pitch và Energy
    f0_contour = compute_f0_autocorr(audio_24k, sample_rate=target_sample_rate)
    voiced_f0 = f0_contour[f0_contour > 0]

    if len(voiced_f0) > 0:
        f0_mean = float(np.mean(voiced_f0))
        f0_median = float(np.median(voiced_f0))
        f0_std = float(np.std(voiced_f0))
        f0_min = float(np.min(voiced_f0))
        f0_max = float(np.max(voiced_f0))
        voiced_ratio = float(len(voiced_f0) / len(f0_contour))
    else:
        f0_mean = 230.0
        f0_median = 230.0
        f0_std = 0.0
        f0_min = 180.0
        f0_max = 350.0
        voiced_ratio = 0.0

    rms_energy = float(np.sqrt(np.mean(audio_24k ** 2)))

    # Phân loại dải cao độ giọng nói
    if f0_median < 135.0:
        gender_band = "Trầm nam (Deep Male / Bass)"
    elif f0_median < 175.0:
        gender_band = "Trung nam / Nữ trầm (Baritone / Contralto)"
    elif f0_median < 230.0:
        gender_band = "Trung nữ / Nam cao (Mezzo-Soprano / Tenor)"
    else:
        gender_band = "Cao nữ / Trẻ em (Soprano / Airy Childish Female)"

    # Đề xuất độ lệch semitones cho speaking voice tự nhiên (~230Hz so với nền ~210Hz)
    speaking_f0 = target_speaking_f0 if target_speaking_f0 is not None else min(f0_median, 235.0)
    baseline_f0 = 210.0 if ("nữ" in gender_band.lower() or f0_median >= 165) else 135.0
    pitch_offset_st = round(12.0 * np.log2(max(speaking_f0, 50.0) / baseline_f0), 1)

    profile = {
        "speaker_name": in_path.stem,
        "source_file": str(in_path),
        "reference_wav": str(out_wav),
        "sample_rate": target_sample_rate,
        "channels": 1,
        "duration_seconds": duration_s,
        "f0_mean_hz": round(f0_mean, 2),
        "f0_median_hz": round(f0_median, 2),
        "f0_std_hz": round(f0_std, 2),
        "f0_min_hz": round(f0_min, 2),
        "f0_max_hz": round(f0_max, 2),
        "target_speaking_f0_hz": round(speaking_f0, 1),
        "voiced_frame_ratio": round(voiced_ratio, 3),
        "rms_energy": round(rms_energy, 4),
        "estimated_pitch_band": gender_band,
        "timbre_description": "Nữ trẻ, mỏng, nhẹ, nhiều hơi (airy soprano), phát âm tự nhiên",
        "recommended_pitch_shift_semitones": pitch_offset_st,
    }
    if ref_text:
        profile["ref_text"] = ref_text
        profile["reference_transcript"] = ref_text

    profile["enrolled_at"] = datetime.now().isoformat()
    profile["status"] = "ready"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    logger.info("=" * 65)
    logger.info("✅ ĐĂNG KÝ GIỌNG NÓI THÀNH CÔNG")
    logger.info("• Tệp âm thanh 24kHz  : %s", out_wav)
    logger.info("• Hồ sơ người nói     : %s", out_json)
    logger.info("• F0 Trung vị         : %.1f Hz (%s)", f0_median, gender_band)
    logger.info("• Thời lượng mẫu      : %.2f giây", duration_s)
    logger.info("=" * 65)

    return audio_24k, profile


def parse_args():
    parser = argparse.ArgumentParser(description="Chuẩn hóa giọng nói và trích xuất hồ sơ speaker profile.")
    parser.add_argument("--input", type=str, required=True, help="Đường dẫn tệp âm thanh hoặc video (.wav, .mp3, .m4a, .mp4)")
    parser.add_argument("--output_wav", type=str, default=str(DEFAULT_OUTPUT_WAV), help="Đường dẫn lưu tệp reference_24k.wav")
    parser.add_argument("--profile_json", type=str, default=str(DEFAULT_PROFILE_JSON), help="Đường dẫn lưu speaker_profile.json")
    return parser.parse_args()


def main():
    args = parse_args()
    enroll_voice(
        input_file=args.input,
        output_wav_path=args.output_wav,
        profile_json_path=args.profile_json,
    )


if __name__ == "__main__":
    main()
