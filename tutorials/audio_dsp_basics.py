"""
Tutorial: Nền tảng Kỹ thuật Tín hiệu Âm thanh số (Audio DSP).
Chương trình thu âm thời gian thực theo frame 100ms và hiển thị thanh đo âm lượng (VU Meter).
"""
import math
import sys
import time
import numpy as np

# Cấu hình UTF-8 cho Terminal Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ==============================================================================
# 1. THAM SỐ TÍN HIỆU ÂM THANH (AUDIO DSP PARAMETERS)
# ==============================================================================
SAMPLE_RATE = 16000           # 16000 Hz: Tiêu chuẩn của các mô hình Voice AI
FRAME_DURATION_MS = 100       # Kích thước mỗi khung: 100 ms (đủ nhạy cho mắt người)
CHUNK_SIZE = int(SAMPLE_RATE * (FRAME_DURATION_MS / 1000.0))  # 1600 mẫu/khung
CHANNELS = 1                  # Mono: 1 kênh âm thanh duy nhất
DTYPE = "float32"             # Dữ liệu số thực chuẩn hóa trong đoạn [-1.0, 1.0]

# Thang đo hiển thị trực quan (Visual Meter)
METER_MIN_DB = -60.0          # Mức im lặng tuyệt đối của phòng (-60 dB)
METER_MAX_DB = -10.0          # Mức nói to gần micro (-10 dB)
BAR_LENGTH = 30               # Độ dài thanh đo hiển thị (ký tự)


def calculate_rms(audio_chunk: np.ndarray) -> float:
    """Tính năng lượng hiệu dụng Root Mean Square (RMS).
    
    Công thức: RMS = sqrt( (1/N) * sum(x_i^2) )
    """
    if audio_chunk.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio_chunk, dtype=np.float64))))


def calculate_dbfs(rms: float) -> float:
    """Chuyển đổi năng lượng RMS sang Decibels Full Scale (dBFS).
    
    Công thức: dBFS = 20 * log10(RMS)
    Thêm epsilon 1e-9 để tránh lỗi toán học log(0).
    """
    if rms < 1e-7:
        return -100.0
    db = 20.0 * math.log10(rms)
    return max(-100.0, min(0.0, db))


def render_vu_meter(db: float, threshold_db: float = -38.0) -> str:
    """Tạo thanh đo đồ họa bằng ký tự terminal."""
    ratio = (db - METER_MIN_DB) / (METER_MAX_DB - METER_MIN_DB)
    ratio = max(0.0, min(1.0, ratio))
    filled_len = int(round(BAR_LENGTH * ratio))

    bar = "#" * filled_len + "-" * (BAR_LENGTH - filled_len)
    status = "SPEAKING" if db > threshold_db else "SILENCE "

    return f"[{bar}] {db:6.1f} dBFS | Status: {status}"


def main():
    print("=" * 70)
    print("🎙️ AUDIO DSP: REAL-TIME RMS & DECIBEL VISUALIZER")
    print(f"Sample Rate: {SAMPLE_RATE} Hz | Chunk: {CHUNK_SIZE} samples ({FRAME_DURATION_MS} ms) | Channels: Mono")
    print("Nói vào micro để quan sát thanh đo phản ứng thời gian thực.")
    print("Nhấn Ctrl + C để dừng chương trình.")
    print("=" * 70)

    try:
        import sounddevice as sd
    except ImportError:
        print("❌ Lỗi: Thư viện 'sounddevice' chưa được cài đặt. Vui lòng chạy: pip install sounddevice")
        return

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            channels=CHANNELS,
            dtype=DTYPE,
        ) as stream:
            while True:
                audio_chunk, overflowed = stream.read(CHUNK_SIZE)
                samples = audio_chunk.flatten()

                rms = calculate_rms(samples)
                db = calculate_dbfs(rms)

                meter_str = render_vu_meter(db, threshold_db=-38.0)
                sys.stdout.write(f"\r{meter_str}")
                sys.stdout.flush()
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nĐã dừng thu âm. Kết thúc chương trình.")
    except Exception as e:
        print(f"\n❌ Lỗi phần cứng âm thanh: {e}")


if __name__ == "__main__":
    main()
