"""
Parrot Bot (Vẹt bắt chước)
Quy trình: Micro -> VAD -> Faster-Whisper (STT) -> In text -> Edge-TTS (TTS) -> Phát ra Loa.
Hoàn toàn không cần LLM.
"""
import asyncio
import io
import os
import sys
import tempfile
import time
from pathlib import Path
import numpy as np

# Cấu hình hiển thị UTF-8 trên Windows Console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Thêm đường dẫn gốc vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import deque


# ==============================================================================
# 1. BỘ PHÁT HIỆN TIẾNG NÓI TỐI GIẢN (ENERGY VAD)
# ==============================================================================
class SimpleVAD:
    """VAD năng lượng tích hợp Pre-buffer 300ms và bộ trễ ngắt câu 1.2s."""

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 512, threshold_db: float = -38.0, silence_timeout_s: float = 1.2):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.threshold_db = threshold_db
        self.chunk_duration_s = chunk_size / sample_rate
        self.max_silence_frames = int(silence_timeout_s / self.chunk_duration_s)
        self.pre_buffer_frames = int(0.3 / self.chunk_duration_s)  # 300ms pre-buffer

        self.pre_buffer: deque[np.ndarray] = deque(maxlen=self.pre_buffer_frames)
        self.recording_buffer: list[np.ndarray] = []
        self.is_recording = False
        self.silence_frames = 0

    def process(self, chunk: np.ndarray) -> tuple[bool, np.ndarray | None]:
        rms = np.sqrt(np.mean(np.square(chunk, dtype=np.float64)))
        db = 20.0 * np.log10(rms) if rms > 1e-7 else -100.0
        is_speech = db > self.threshold_db

        if not self.is_recording:
            self.pre_buffer.append(chunk)
            if is_speech:
                self.is_recording = True
                self.recording_buffer.extend(list(self.pre_buffer))
                self.pre_buffer.clear()
                self.silence_frames = 0
            return False, None
        else:
            self.recording_buffer.append(chunk)
            if is_speech:
                self.silence_frames = 0
            else:
                self.silence_frames += 1
                if self.silence_frames >= self.max_silence_frames:
                    # Chốt câu hoàn chỉnh
                    audio = np.concatenate(self.recording_buffer)
                    self.recording_buffer.clear()
                    self.pre_buffer.clear()
                    self.is_recording = False
                    self.silence_frames = 0
                    return True, audio
            return False, None


# ==============================================================================
# 2. KHỞI TẠO MÔ HÌNH SPEECH-TO-TEXT (FASTER-WHISPER)
# ==============================================================================
def load_stt_model(model_size: str = "base"):
    """Tải mô hình Faster-Whisper chạy trên CPU với định dạng int8 lượng tử hóa."""
    try:
        from faster_whisper import WhisperModel
        print(f"🔄 Đang nạp mô hình Faster-Whisper ({model_size}) vào CPU (int8 quantization)...")
        # compute_type='int8' giúp mô hình chạy siêu nhanh và tiết kiệm RAM trên CPU
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("✅ Mô hình Faster-Whisper đã sẵn sàng!")
        return model
    except ImportError:
        print("⚠️ Chưa cài đặt 'faster-whisper'. Chạy: pip install faster-whisper")
        return None


def transcribe_audio(model, audio_float32: np.ndarray, sample_rate: int = 16000) -> str:
    """Chuyển đổi trực tiếp mảng NumPy float32 thành text tiếng Việt mà KHÔNG cần ghi file đĩa."""
    if model is None:
        return "Xin chào, đây là bản giả lập khi chưa cài faster-whisper"

    # Faster-Whisper nhận trực tiếp mảng 1D float32 trong khoảng [-1.0, 1.0]
    segments, info = model.transcribe(
        audio_float32,
        language="vi",      # Khóa ngôn ngữ Tiếng Việt để tránh nhận diện nhầm
        beam_size=1,        # beam_size=1 (Greedy search) tối ưu độ trễ thấp nhất
        vad_filter=False,   # Đã có VAD tầng ngoài xử lý cắt khoảng lặng
    )
    text = " ".join([seg.text.strip() for seg in segments]).strip()
    return text


# ==============================================================================
# 3. TEXT-TO-SPEECH (EDGE-TTS) & PHÁT ÂM THANH
# ==============================================================================
async def speak_text_edge(text: str, voice: str = "vi-VN-NamMinhNeural") -> None:
    """Tổng hợp văn bản tiếng Việt thành giọng nói tự nhiên và phát ra loa."""
    if not text.strip():
        return

    try:
        import edge_tts
        import sounddevice as sd
        import av  # PyAV được cài tự động khi cài faster-whisper
    except ImportError:
        # Fallback dùng Windows PowerShell speech nếu chưa cài đủ thư viện
        print(f"🤖 [Loa phát]: \"{text}\"")
        try:
            import subprocess
            cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\')"'
            subprocess.run(cmd, shell=True)
        except Exception:
            pass
        return

    # 1. Gọi Edge-TTS lấy stream MP3
    communicate = edge_tts.Communicate(text=text, voice=voice)
    mp3_buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_buffer.write(chunk["data"])

    mp3_bytes = mp3_buffer.getvalue()
    if not mp3_bytes:
        return

    # 2. Giải mã MP3 bytes thành mảng NumPy float32 bằng PyAV
    try:
        container = av.open(io.BytesIO(mp3_bytes))
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        audio_frames = []
        for frame in container.decode(audio=0):
            frame.pts = None
            for resampled_frame in resampler.resample(frame):
                audio_frames.append(resampled_frame.to_ndarray()[0])

        if audio_frames:
            audio_array = np.concatenate(audio_frames).astype(np.float32)
            # 3. Phát trực tiếp ra loa bằng sounddevice (chặn cho tới khi phát xong)
            sd.play(audio_array, samplerate=16000)
            sd.wait()
    except Exception as e:
        # Fallback lưu file tạm nếu PyAV gặp lỗi giải mã
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name
            f.write(mp3_bytes)
        os.system(f'start "" /min wmplayer "{temp_path}"')


# ==============================================================================
# 4. CHƯƠNG TRÌNH PARROT BOT CHÍNH (MAIN LOOP)
# ==============================================================================
def main():
    print("=" * 70)
    print("🦜 PARROT BOT: VẸT BẮT CHƯỚC THỜI GIAN THỰC")
    print("Quy trình: Micro -> VAD -> Faster-Whisper (STT) -> Edge-TTS -> Loa")
    print("Nhấn Ctrl + C để dừng bot.")
    print("=" * 70)

    try:
        import sounddevice as sd
    except ImportError:
        print("❌ Chưa cài đặt sounddevice. Chạy: pip install sounddevice")
        return

    # Tải mô hình STT
    stt_model = load_stt_model(model_size="base")

    sample_rate = 16000
    chunk_size = 512  # 32ms
    vad = SimpleVAD(sample_rate=sample_rate, chunk_size=chunk_size, threshold_db=-38.0, silence_timeout_s=1.2)

    print("\n🟢 Parrot Bot đang lắng nghe! Hãy nói một câu vào micro...")

    with sd.InputStream(samplerate=sample_rate, blocksize=chunk_size, channels=1, dtype="float32") as stream:
        while True:
            try:
                frame, _ = stream.read(chunk_size)
                finished, audio_data = vad.process(frame.flatten())

                if vad.is_recording and vad.silence_frames == 0:
                    sys.stdout.write("\r🔴 Đang thu âm tiếng bạn nói...           ")
                    sys.stdout.flush()

                if finished and audio_data is not None:
                    sys.stdout.write("\r⏳ Đang nhận dạng giọng nói (STT)...      \n")
                    sys.stdout.flush()

                    t0 = time.perf_counter()
                    user_text = transcribe_audio(stt_model, audio_data, sample_rate)
                    stt_time = (time.perf_counter() - t0) * 1000

                    if user_text.strip():
                        print(f"👤 Bạn vừa nói : \"{user_text}\" (STT mất {stt_time:.0f}ms)")
                        print(f"🦜 Vẹt đọc lại : \"{user_text}\"")

                        # Gọi TTS đọc lại câu nói
                        asyncio.run(speak_text_edge(user_text))
                    else:
                        print("⚠️ Không nhận dạng được từ nào (có thể là tạp âm).")

                    print("\n🟢 Tiếp tục lắng nghe...")

            except KeyboardInterrupt:
                print("\n\n👋 Tạm biệt! Đã dừng Parrot Bot.")
                break


if __name__ == "__main__":
    main()
