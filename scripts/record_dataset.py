"""Interactive script to record speech dataset samples with transcripts for training/fine-tuning."""
import json
import os
import sys
import time
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

from config.audio_config import AudioConfig
from core.audio_recorder import AudioRecorder

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
METADATA_FILE = PROCESSED_DATA_DIR / "metadata.jsonl"


def record_sample(prompt_text: str, index: int, config: AudioConfig) -> None:
    """Record a single speech sample for the given prompt text."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"[{index + 1}] Xin vui lòng đọc câu sau:")
    print(f"👉 \"{prompt_text}\"")
    print("=" * 60)
    input("Nhấn [Enter] để BẮT ĐẦU thu âm...")

    recorder = AudioRecorder(config)
    recorder.start()
    frames = []

    print("🔴 ĐANG THU ÂM... (Nhấn [Enter] để KẾT THÚC)")
    start_time = time.time()

    # Capture in background until user hits enter
    import threading
    stop_event = threading.Event()

    def wait_enter():
        input()
        stop_event.set()

    t = threading.Thread(target=wait_enter)
    t.daemon = True
    t.start()

    while not stop_event.is_set():
        frame = recorder.read_frame(timeout=0.05)
        if frame is not None:
            frames.append(frame)

    recorder.stop()
    duration = time.time() - start_time

    if not frames:
        print("❌ Không thu được dữ liệu âm thanh. Bỏ qua.")
        return

    audio = np.concatenate(frames)
    filename = f"sample_{int(time.time())}_{index:04d}.wav"
    raw_path = RAW_DATA_DIR / filename

    # Save 16-bit PCM WAV
    int16_audio = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    try:
        import scipy.io.wavfile as wavfile
        wavfile.write(str(raw_path), config.sample_rate, int16_audio)
    except ImportError:
        with wave.open(str(raw_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(config.sample_rate)
            wf.writeframes(int16_audio.tobytes())

    record = {
        "audio_filepath": str(raw_path).replace("\\", "/"),
        "text": prompt_text,
        "duration": round(duration, 3),
        "sample_rate": config.sample_rate,
    }

    with open(METADATA_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ Đã lưu: {raw_path} ({duration:.2f}s)")


def main():
    config = AudioConfig()
    sample_prompts = [
        "Xin chào, thời tiết hôm nay thế nào?",
        "Bật đèn phòng khách và mở nhạc nhẹ giúp tôi.",
        "Hôm nay là ngày mấy trong tuần?",
        "Nhắc tôi uống thuốc lúc tám giờ tối.",
        "Tạm biệt và hẹn gặp lại bạn sau.",
    ]

    print("🎙️ CÔNG CỤ THU ÂM MẪU DỮ LIỆU GIỌNG NÓI")
    print("---------------------------------------")

    for i, prompt in enumerate(sample_prompts):
        try:
            record_sample(prompt, i, config)
        except KeyboardInterrupt:
            print("\nĐã dừng tiến trình thu âm.")
            break

    print(f"\nHoàn tất! Metadata được lưu tại: {METADATA_FILE}")


if __name__ == "__main__":
    main()
