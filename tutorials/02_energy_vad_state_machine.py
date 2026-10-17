"""
Tutorial 02: Voice Activity Detection (VAD) dựa trên Năng lượng và Máy trạng thái (Finite State Machine).
Bao gồm:
- 3 Trạng thái: LISTENING -> RECORDING <-> SILENCE_DETECTION
- Pre-buffer 300ms bảo toàn phụ âm đầu câu
- Tự động ngắt và trả về mảng float32 khi im lặng vượt quá 1.5 giây.
"""
import enum
import math
import sys
from collections import deque
from typing import Optional
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class VADState(enum.Enum):
    LISTENING = "LISTENING"                     # Chờ tiếng nói vượt ngưỡng
    RECORDING = "RECORDING"                     # Đang ghi nhận tiếng nói liên tục
    SILENCE_DETECTION = "SILENCE_DETECTION"     # Đang theo dõi khoảng lặng (Hangover)


class EnergyVAD:
    """Bộ phát hiện tiếng nói (VAD) dựa trên năng lượng RMS với Pre-Buffer và Máy trạng thái."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 30,          # Khung chuẩn: 30ms (~480 samples @ 16kHz)
        speech_threshold_db: float = -38.0,   # Ngưỡng dB bắt đầu nói
        silence_timeout_s: float = 1.5,       # Thời gian im lặng tối đa trước khi chốt câu (1.5s)
        pre_buffer_duration_ms: int = 300,    # Bộ đệm giữ 300ms phụ âm đầu câu
        min_speech_duration_ms: int = 90,     # Phải vượt ngưỡng liên tục >= 90ms để kích hoạt
    ):
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(self.sample_rate * (self.chunk_duration_ms / 1000.0))
        self.speech_threshold_db = speech_threshold_db
        self.silence_timeout_s = silence_timeout_s

        # Số khung tương ứng với các khoảng thời gian
        self.min_speech_frames = max(1, int(min_speech_duration_ms / chunk_duration_ms))
        self.max_silence_frames = max(1, int((silence_timeout_s * 1000.0) / chunk_duration_ms))
        pre_buffer_frames = max(1, int(pre_buffer_duration_ms / chunk_duration_ms))

        # 1. Quản lý Bộ đệm (Buffer Management)
        # Pre-buffer dùng deque với độ dài cố định: tự động đẩy frame cũ ra khi đầy
        self.pre_buffer: deque[np.ndarray] = deque(maxlen=pre_buffer_frames)
        # Buffer chính gom dữ liệu của toàn bộ câu nói
        self.recording_buffer: list[np.ndarray] = []

        # 2. Quản lý Trạng thái & Bộ đếm (State & Counters)
        self.state = VADState.LISTENING
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0

    @staticmethod
    def calculate_rms_db(chunk: np.ndarray) -> float:
        """Tính năng lượng dBFS của frame âm thanh."""
        if chunk.size == 0:
            return -100.0
        rms = np.sqrt(np.mean(np.square(chunk, dtype=np.float64)))
        if rms < 1e-7:
            return -100.0
        return float(20.0 * math.log10(rms))

    def process_frame(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """Xử lý từng khối audio (ví dụ 30ms hoặc 100ms).
        
        Args:
            audio_chunk: Mảng NumPy 1D (float32).
            
        Returns:
            np.ndarray (float32): Trả về toàn bộ câu hoàn chỉnh nếu phát hiện kết thúc câu.
            None: Nếu đang trong quá trình nghe hoặc đang nói dở.
        """
        flat_chunk = audio_chunk.flatten().astype(np.float32)
        rms_db = self.calculate_rms_db(flat_chunk)
        is_above_threshold = rms_db > self.speech_threshold_db

        # ----------------------------------------------------------------------
        # TRẠNG THÁI 1: LẮNG NGHE (LISTENING)
        # ----------------------------------------------------------------------
        if self.state == VADState.LISTENING:
            # Liên tục lưu frame vào Pre-buffer xoay vòng
            self.pre_buffer.append(flat_chunk)

            if is_above_threshold:
                self.consecutive_speech_frames += 1
                # Nếu vượt ngưỡng đủ lâu (tránh tiếng click chuột hay xung ngắn)
                if self.consecutive_speech_frames >= self.min_speech_frames:
                    # Chuyển sang RECORDING
                    self.state = VADState.RECORDING
                    self.consecutive_speech_frames = 0
                    self.consecutive_silence_frames = 0

                    # 1. Đổ toàn bộ Pre-buffer (300ms) vào buffer ghi âm chính
                    self.recording_buffer.extend(list(self.pre_buffer))
                    # 2. Xóa pre-buffer để sẵn sàng cho lần sau
                    self.pre_buffer.clear()
            else:
                self.consecutive_speech_frames = 0

            return None

        # ----------------------------------------------------------------------
        # TRẠNG THÁI 2: ĐANG GHI (RECORDING)
        # ----------------------------------------------------------------------
        elif self.state == VADState.RECORDING:
            self.recording_buffer.append(flat_chunk)

            if is_above_threshold:
                # Vẫn đang nói, reset bộ đếm im lặng
                self.consecutive_silence_frames = 0
            else:
                # Bắt đầu phát hiện im lặng -> Chuyển sang kiểm tra khoảng lặng
                self.consecutive_silence_frames += 1
                self.state = VADState.SILENCE_DETECTION

            return None

        # ----------------------------------------------------------------------
        # TRẠNG THÁI 3: THEO DÕI IM LẶNG (SILENCE DETECTION / HANGOVER)
        # ----------------------------------------------------------------------
        elif self.state == VADState.SILENCE_DETECTION:
            # QUAN TRỌNG: Vẫn tiếp tục ghi các frame im lặng này vào buffer
            # để câu nói không bị ngắt đuôi đột ngột
            self.recording_buffer.append(flat_chunk)

            if is_above_threshold:
                # Người dùng nói tiếp (chỉ là tạm dừng lấy hơi)! Quay lại RECORDING
                self.state = VADState.RECORDING
                self.consecutive_silence_frames = 0
            else:
                self.consecutive_silence_frames += 1

                # Nếu thời gian im lặng vượt quá 1.5 giây
                if self.consecutive_silence_frames >= self.max_silence_frames:
                    # Chốt câu nói! Ghép toàn bộ mảng buffer thành một mảng NumPy duy nhất
                    complete_utterance = np.concatenate(self.recording_buffer)

                    # Reset toàn bộ trạng thái về LISTENING
                    self.reset()

                    # Trả về mảng âm thanh hoàn chỉnh
                    return complete_utterance

            return None

        return None

    def reset(self) -> None:
        """Đưa máy trạng thái về vị trí ban đầu."""
        self.state = VADState.LISTENING
        self.pre_buffer.clear()
        self.recording_buffer.clear()
        self.consecutive_speech_frames = 0
        self.consecutive_silence_frames = 0


# ==============================================================================
# DEMO MÔ PHỎNG KIỂM THỬ THUẬT TOÁN (SYNTHETIC SIMULATION TEST)
# ==============================================================================
if __name__ == "__main__":
    sr = 16000
    chunk_ms = 30
    chunk_samples = int(sr * (chunk_ms / 1000.0))  # 480 samples

    vad = EnergyVAD(
        sample_rate=sr,
        chunk_duration_ms=chunk_ms,
        speech_threshold_db=-38.0,
        silence_timeout_s=1.5,
        pre_buffer_duration_ms=300,
    )

    print("=" * 65)
    print("🧪 MÔ PHỎNG CHU TRÌNH PHÁT HIỆN TIẾNG NÓI VỚI ENERGY VAD")
    print("=" * 65)

    # 1. Mô phỏng 1.0 giây im lặng phòng (Nhiễu nền nhẹ -55 dBFS)
    silence_frame = np.random.normal(0, 0.001, chunk_samples).astype(np.float32)
    # 2. Mô phỏng 1.5 giây người dùng nói: "Xin chào thời tiết hôm nay thế nào" (-20 dBFS)
    t = np.linspace(0, chunk_ms / 1000.0, chunk_samples, endpoint=False)
    speech_frame = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    total_frames = []
    # 1.0s im lặng (33 frames)
    total_frames.extend([(silence_frame, "Im lặng phòng") for _ in range(33)])
    # 1.0s nói (33 frames)
    total_frames.extend([(speech_frame, "Người đang nói...") for _ in range(33)])
    # 0.3s tạm ngưng lấy hơi (10 frames)
    total_frames.extend([(silence_frame, "Tạm ngưng 300ms...") for _ in range(10)])
    # 0.8s nói tiếp (26 frames)
    total_frames.extend([(speech_frame, "Nói tiếp...") for _ in range(26)])
    # 1.8s im lặng kết thúc câu (60 frames > 1.5s timeout)
    total_frames.extend([(silence_frame, "Im lặng kết thúc...") for _ in range(60)])

    detected_audio = None
    for idx, (frame, desc) in enumerate(total_frames):
        current_state_before = vad.state.value
        result = vad.process_frame(frame)

        if vad.state.value != current_state_before:
            print(f"Frame {idx:03d} ({(idx*chunk_ms)/1000.0:.2f}s): State [{current_state_before}] -> [{vad.state.value}] | ({desc})")

        if result is not None:
            detected_audio = result
            print("=" * 65)
            print(f"🎉 SỰ KIỆN KẾT THÚC CÂU (SPEECH_END) ĐÃ ĐƯỢC KÍCH HOẠT!")
            print(f"• Tổng số mẫu audio thu được: {len(detected_audio)} samples")
            print(f"• Tổng thời lượng câu:        {len(detected_audio) / sr:.2f} giây")
            print(f"• Kiểu dữ liệu mảng:         {detected_audio.dtype}")
            print("=" * 65)

    assert detected_audio is not None, "VAD phải bắt được câu nói hoàn chỉnh!"
    print("✅ Kiểm thử mô phỏng thành công 100%!")
