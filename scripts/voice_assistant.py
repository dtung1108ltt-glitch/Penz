"""
Voice Assistant: Trợ lý giọng nói hoàn chỉnh tích hợp não bộ Ollama LLM.
Pipeline: Micro -> VAD -> Faster-Whisper (STT) -> Ollama (LLM) -> Edge-TTS (TTS) -> Loa.
"""
import asyncio
import io
import os
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
import numpy as np
import requests

# Cấu hình UTF-8 console Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Thêm đường dẫn gốc vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG & OLLAMA
# ==============================================================================
SAMPLE_RATE = 16000
CHUNK_SIZE = 512              # 32ms mỗi frame
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")  # hoặc llama3.2:3b

# System prompt tối ưu cho giọng nói: Ngắn gọn, không markdown
SYSTEM_PROMPT = (
    "Bạn là một người bạn thân thiết trong cuộc hội thoại giọng nói. "
    "Hãy trò chuyện tự nhiên, gần gũi bằng tiếng Việt. "
    "Câu trả lời bắt buộc dưới 30 từ. "
    "Tuyệt đối không dùng ký hiệu markdown như *, #, -, danh sách hay gạch đầu dòng."
)


# ==============================================================================
# 2. BỘ PHÁT HIỆN TIẾNG NÓI (ENERGY VAD VỚI PRE-BUFFER)
# ==============================================================================
class SimpleVAD:
    """VAD năng lượng: Pre-buffer 300ms, tự động chốt câu khi im lặng 1.2s."""

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 512, threshold_db: float = -38.0, silence_timeout_s: float = 1.2):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.threshold_db = threshold_db
        self.chunk_duration_s = chunk_size / sample_rate
        self.max_silence_frames = int(silence_timeout_s / self.chunk_duration_s)
        self.pre_buffer_frames = int(0.3 / self.chunk_duration_s)

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
                    audio = np.concatenate(self.recording_buffer)
                    self.recording_buffer.clear()
                    self.pre_buffer.clear()
                    self.is_recording = False
                    self.silence_frames = 0
                    return True, audio
            return False, None


# ==============================================================================
# 3. SPEECH-TO-TEXT (FASTER-WHISPER)
# ==============================================================================
def load_stt_model(model_size: str = "base"):
    try:
        from faster_whisper import WhisperModel
        print(f"🔄 Đang nạp mô hình STT Faster-Whisper ({model_size}) trên CPU...")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        print("✅ Mô hình STT đã sẵn sàng!")
        return model
    except ImportError:
        print("⚠️ Chưa cài đặt 'faster-whisper'. Chế độ giả lập kích hoạt.")
        return None


def transcribe_audio(model, audio_float32: np.ndarray) -> str:
    if model is None:
        return "Thời tiết hôm nay thế nào bạn ơi?"

    segments, info = model.transcribe(
        audio_float32,
        language="vi",
        beam_size=1,
        vad_filter=False,
    )
    return " ".join([seg.text.strip() for seg in segments]).strip()


# ==============================================================================
# 4. NÃO BỘ LLM (OLLAMA CLIENT CÓ XỬ LÝ LỖI & CONTEXT)
# ==============================================================================
def query_ollama(user_text: str, history: list[dict[str, str]], timeout_s: float = 8.0) -> str:
    """Gửi tin nhắn sang Ollama API kèm ngữ cảnh hội thoại và xử lý lỗi mạng."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Thêm 4 lượt thoại gần nhất để nhớ ngữ cảnh ngắn
    messages.extend(history[-4:])
    messages.append({"role": "user", "content": user_text})

    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 60,  # Giới hạn số token sinh ra để phản hồi siêu nhanh
                },
            },
            timeout=timeout_s,
        )

        if response.status_code == 200:
            result = response.json()
            reply = result.get("message", {}).get("content", "").strip()
            # Làm sạch ký tự markdown nếu mô hình vô tình sinh ra
            reply = reply.replace("*", "").replace("#", "").replace("- ", "").replace("\n", " ")
            return reply if reply else "Tôi nghe bạn rồi, bạn nói tiếp đi."
        else:
            print(f"\n⚠️ Ollama API trả về mã lỗi: {response.status_code}")
            return "Tôi đang bị quá tải một chút, bạn nói lại nhé."

    except requests.exceptions.ConnectionError:
        print(f"\n❌ Không thể kết nối tới Ollama tại {OLLAMA_HOST}. Hãy kiểm tra xem Ollama đã chạy chưa (`ollama serve`).")
        return "Xin lỗi bạn, tôi chưa kết nối được với não bộ Ollama."
    except requests.exceptions.Timeout:
        print("\n⏳ Ollama phản hồi quá lâu (Timeout).")
        return "Tôi suy nghĩ hơi lâu một chút, bạn hỏi lại câu ngắn hơn nhé."
    except Exception as e:
        print(f"\n⚠️ Lỗi phát sinh từ LLM: {e}")
        return "Tôi vừa gặp sự cố nhỏ, chúng ta tiếp tục nhé."


# ==============================================================================
# 5. TEXT-TO-SPEECH (EDGE-TTS & PHÁT ÂM THANH)
# ==============================================================================
async def speak_text(text: str, voice: str = "vi-VN-NamMinhNeural") -> None:
    if not text.strip():
        return

    try:
        import edge_tts
        import sounddevice as sd
        import av

        communicate = edge_tts.Communicate(text=text, voice=voice)
        mp3_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_buffer.write(chunk["data"])

        container = av.open(io.BytesIO(mp3_buffer.getvalue()))
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
        audio_frames = []
        for frame in container.decode(audio=0):
            frame.pts = None
            for resampled in resampler.resample(frame):
                audio_frames.append(resampled.to_ndarray()[0])

        if audio_frames:
            audio_array = np.concatenate(audio_frames).astype(np.float32)
            sd.play(audio_array, samplerate=16000)
            sd.wait()

    except Exception:
        # Fallback qua PowerShell nếu chưa có đủ thư viện audio
        print(f"🤖 [Loa]: \"{text}\"")
        try:
            import subprocess
            cmd = f'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\')"'
            subprocess.run(cmd, shell=True)
        except Exception:
            pass


# ==============================================================================
# 6. VÒNG LẶP ĐIỀU PHỐI CHÍNH (FULL PIPELINE WITH RESILIENCE)
# ==============================================================================
def run_voice_assistant():
    print("=" * 70)
    print("🧠 VOICE AI ASSISTANT: TÍCH HỢP NÃO BỘ OLLAMA")
    print(f"Pipeline : Micro -> VAD -> Whisper -> Ollama ({OLLAMA_MODEL}) -> TTS -> Loa")
    print("Nhấn Ctrl + C để dừng trợ lý.")
    print("=" * 70)

    try:
        import sounddevice as sd
    except ImportError:
        print("❌ Cần cài đặt sounddevice: pip install sounddevice")
        return

    stt_model = load_stt_model(model_size="base")
    vad = SimpleVAD(sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE, threshold_db=-38.0, silence_timeout_s=1.2)
    conversation_history: list[dict[str, str]] = []

    print("\n🟢 Trợ lý AI đang lắng nghe... Hãy nói điều gì đó!")

    # Vòng lặp bao bọc bảo vệ khi phần cứng micro hoặc stream gặp sự cố
    while True:
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE, channels=1, dtype="float32") as stream:
                while True:
                    try:
                        frame, _ = stream.read(CHUNK_SIZE)
                        finished, audio_data = vad.process(frame.flatten())

                        if vad.is_recording and vad.silence_frames == 0:
                            sys.stdout.write("\r🔴 Đang lắng nghe giọng bạn...          ")
                            sys.stdout.flush()

                        if finished and audio_data is not None:
                            sys.stdout.write("\r⏳ [1/3] Đang chuyển giọng nói thành chữ (STT)...   \n")
                            sys.stdout.flush()

                            # 1. Speech-to-Text
                            t0 = time.perf_counter()
                            user_text = transcribe_audio(stt_model, audio_data)
                            stt_time = (time.perf_counter() - t0) * 1000

                            if not user_text.strip():
                                print("⚠️ Không nghe rõ từ nào, mời bạn nói lại.")
                                continue

                            print(f"👤 Bạn      : \"{user_text}\" (STT: {stt_time:.0f}ms)")

                            # 2. Suy luận não bộ qua LLM Ollama
                            sys.stdout.write("🧠 [2/3] Não bộ AI đang suy nghĩ (Ollama)...        \n")
                            sys.stdout.flush()
                            t1 = time.perf_counter()
                            ai_reply = query_ollama(user_text, conversation_history)
                            llm_time = (time.perf_counter() - t1) * 1000

                            print(f"🤖 Trợ lý AI: \"{ai_reply}\" (LLM: {llm_time:.0f}ms)")

                            # Cập nhật lịch sử đối thoại ngắn
                            conversation_history.append({"role": "user", "content": user_text})
                            conversation_history.append({"role": "assistant", "content": ai_reply})

                            # 3. Text-to-Speech phát ra loa
                            sys.stdout.write("🔊 [3/3] Đang phát âm phản hồi ra loa (TTS)...     \n")
                            sys.stdout.flush()
                            asyncio.run(speak_text(ai_reply))

                            print("\n🟢 Tiếp tục lắng nghe...")

                    except Exception as loop_error:
                        print(f"\n⚠️ Lỗi tạm thời trong chu trình xử lý: {loop_error}")
                        time.sleep(0.5)
                        break  # Thoát stream để khởi động lại nếu lỗi phần cứng micro

        except KeyboardInterrupt:
            print("\n\n👋 Tạm biệt! Đã tắt Trợ lý Voice AI.")
            break
        except Exception as hardware_error:
            print(f"\n❌ Lỗi kết nối thiết bị âm thanh (Micro/Loa): {hardware_error}")
            print("Đang thử kết nối lại micro sau 2 giây...")
            time.sleep(2.0)


if __name__ == "__main__":
    run_voice_assistant()
