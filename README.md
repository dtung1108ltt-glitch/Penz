# 🎙️ Production-Grade Voice AI Architecture & System

Hệ thống Trợ lý Giọng nói Đàm thoại Thời gian thực (Real-Time Conversational Voice AI System) chuẩn cấp độ sản xuất từ con số 0.

Kiến trúc được thiết kế tối ưu hóa độ trễ cực thấp (Ultra Low-Latency < 1.2s Voice-to-Voice), hỗ trợ ngắt lời tự nhiên (**Barge-in / Interruption**), và phân tách module triệt để theo các nguyên lý công nghệ chuẩn công nghiệp.

---

## 🏛️ Ba Nguyên Tắc Kiến Trúc Cốt Lõi

1. **`core/` hoàn toàn độc lập với AI:**
   - Tầng này chỉ làm việc với tín hiệu số thô (`numpy.ndarray`, $16\text{ kHz}$, tính toán công suất RMS, ZCR, SoundDevice I/O stream, Ring buffer).
   - Nếu toàn bộ mô hình AI bị lỗi hoặc tắt mạng, tầng `core/` vẫn đọc micro và phát loa bình thường mà không bị crash.
2. **`engines/` tuân thủ Dependency Inversion (DIP):**
   - Mọi inference engine đều kế thừa từ `base_engine.py` (`BaseSTTEngine`, `BaseLLMEngine`, `BaseTTSEngine`).
   - Bạn có thể chuyển đổi linh hoạt giữa Edge-TTS sang Piper TTS cục bộ, hoặc Whisper sang Conformer chỉ bằng cách tạo class mới mà không cần chỉnh sửa bất kỳ dòng code nào trong tầng `pipeline/`.
3. **Tách biệt tuyệt đối giữa `training/` và `pipeline/`:**
   - Thư mục `training/` chứa toàn bộ code PyTorch huấn luyện nặng (sử dụng khi build/fine-tune model ban đầu).
   - Khi chạy ứng dụng thời gian thực (`main.py`), hệ thống chỉ import tầng `engines/` và `pipeline/`, giúp tiết kiệm RAM/VRAM và giữ thời gian khởi động ở mức tức thì.

---

## 📁 Cấu Trúc Cây Thư Mục Chi Tiết

```plaintext
voice_ai_system/
├── .env.example                     # Biến môi trường mẫu (HF_TOKEN, OLLAMA_HOST, DEVICE)
├── .gitignore                       # Bỏ qua checkpoints (*.pth, *.bin), cache, venv
├── requirements.txt                 # Khai báo thư viện phụ thuộc chính xác
├── setup.py                         # Đóng gói dự án thành Python package nội bộ
├── README.md                        # Tài liệu hướng dẫn thiết lập & chạy thử nghiệm
├── main.py                          # Entry point khởi chạy ứng dụng Voice Assistant
│
├── config/                          # Quản lý cấu hình tập trung
│   ├── __init__.py
│   ├── audio_config.py              # Tham số phần cứng: sample_rate (16kHz), chunk_size, channels
│   ├── model_config.py              # Tên/đường dẫn model: Whisper size, LLM context length, TTS voice
│   └── vad_config.py                # Ngưỡng phát hiện tiếng nói: silence_thresh, buffer_duration
│
├── data/                            # Lưu trữ dữ liệu huấn luyện & benchmark
│   ├── raw/                         # File ghi âm thô (.wav, .flac)
│   ├── processed/                   # Audio đã chuẩn hóa (16kHz mono) + nhãn transcript (.jsonl)
│   └── synthetic/                   # Dữ liệu đối thoại giả lập phục vụ fine-tuning LLM
│
├── scripts/                         # Kịch bản tiền xử lý & công cụ tiện ích
│   ├── __init__.py
│   ├── record_dataset.py            # Tool thu âm mẫu giọng nói có gán nhãn
│   ├── preprocess_audio.py          # Chuyển đổi định dạng, khử nhiễu băng thông, cắt khoảng lặng
│   └── benchmark_latency.py         # Đo đạc chi tiết Time-to-First-Byte (TTFB) của từng module
│
├── training/                        # Pipeline huấn luyện & Fine-tuning mô hình từ đầu
│   ├── __init__.py
│   ├── datasets.py                  # PyTorch Custom Dataset & Dataloaders cho Audio/Text
│   ├── stt_finetune/                # Tinh chỉnh mô hình nhận dạng tiếng nói (Whisper/Conformer)
│   │   ├── __init__.py
│   │   ├── train_stt.py             # Training loop, loss CTC / Seq2Seq CrossEntropy
│   │   └── evaluate_wer.py          # Đánh giá tỷ lệ lỗi từ (Word Error Rate - WER)
│   ├── llm_finetune/                # Tinh chỉnh LLM hội thoại ngắn (QLoRA)
│   │   ├── __init__.py
│   │   ├── prepare_data.py          # Chuẩn hóa format chat template (ShareGPT/Alpaca)
│   │   └── train_lora.py            # Huấn luyện LoRA adapter bằng PEFT / Unsloth
│   └── tts_train/                   # Huấn luyện/Fine-tune giọng đọc (VITS/Piper/Kokoro)
│       ├── __init__.py
│       ├── extract_features.py      # Trích xuất Mel-spectrogram & F0 pitch
│       └── train_vocoder.py         # Huấn luyện generator & discriminator
│
├── core/                            # Tầng xử lý tín hiệu âm thanh số (DSP & Audio I/O)
│   ├── __init__.py
│   ├── audio_recorder.py            # Stream thu âm non-blocking từ Micro qua SoundDevice
│   ├── audio_player.py              # Stream phát âm thanh ra Loa (hỗ trợ non-blocking & ngắt buffer)
│   ├── ring_buffer.py               # Hàng đợi bộ nhớ đệm tròn (Pre-buffer giữ phụ âm đầu câu)
│   └── vad_detector.py              # Thuật toán ngắt câu: Energy RMS, Zero-Crossing Rate & Silero VAD
│
├── engines/                         # Tầng tích hợp các mô hình suy luận (Inference Layer)
│   ├── __init__.py
│   ├── base_engine.py               # Abstract Base Class (ABC) định nghĩa interface chuẩn
│   ├── stt_engine.py                # Wrapper Whisper/Faster-Whisper (Numpy Audio array -> Text)
│   ├── llm_engine.py                # Streaming LLM Client (Tokens stream -> ngắt câu theo dấu câu)
│   └── tts_engine.py                # Chuyển Text thành mảng Audio bytes (Edge-TTS / Local Piper)
│
├── pipeline/                        # Tầng điều phối hệ thống (System Orchestrator)
│   ├── __init__.py
│   ├── event_bus.py                 # Quản lý tín hiệu bất đồng bộ: USER_SPEAKING, AI_TALKING, INTERRUPT
│   ├── session_manager.py           # Quản lý Context hội thoại ngắn hạn (Conversation buffer)
│   └── orchestrator.py              # Vòng lặp chính: Điều phối Audio -> VAD -> STT -> LLM -> TTS -> Loa
│
└── tests/                           # Unit tests & Integration tests
    ├── __init__.py
    ├── test_audio_hardware.py       # Kiểm tra quyền truy cập Micro & Loa
    ├── test_vad_accuracy.py         # Kiểm tra độ nhạy bắt tiếng nói
    ├── test_stt_transcription.py    # Kiểm tra nhận diện câu mẫu
    └── test_end_to_end.py           # Kiểm tra toàn bộ chu trình với file audio giả lập
```

---

## ⚡ Hướng Dẫn Cài Đặt & Sử Dụng

### 1. Khởi tạo môi trường ảo

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt trên Windows:
venv\Scripts\activate
# Kích hoạt trên Linux/macOS:
source venv/bin/activate
```

### 2. Cài đặt thư viện phụ thuộc

```bash
# Cài đặt các gói core & runtime:
pip install -r requirements.txt

# Cài đặt dự án ở chế độ editable:
pip install -e .
```

### 3. Cấu hình biến môi trường

Sao chép file `.env.example` thành `.env` và điều chỉnh thông số nếu cần:
```bash
copy .env.example .env
```

---

## 🚀 Khởi Chạy Ứng Dụng

### Chế độ Mock Test (Không cần tải model lớn, chạy ngay lập tức):
```bash
python main.py --mock
```

### Chế độ Sản Xuất Toàn Diện (Full Production):
Đảm bảo đã chạy Ollama (`ollama run llama3:8b-instruct-q4_K_M`):
```bash
python main.py --stt faster-whisper --llm ollama --tts edge-tts --device cuda
```

### Đo đạc độ trễ chi tiết (Latency Benchmark):
```bash
python scripts/benchmark_latency.py
```

### Chạy toàn bộ Unit Tests & Integration Tests:
```bash
pytest -v tests/
```

---

## 🔬 Cơ Chế Hoạt Động & Tối Ưu Độ Trễ

1. **Pre-Buffering với `RingBuffer` (300ms):**
   Khi người dùng bắt đầu nói một từ có phụ âm bật (ví dụ: *P*, *T*, *K*, *S*), năng lượng âm thanh ban đầu có thể chưa vượt ngưỡng VAD. `RingBuffer` liên tục lưu 300ms gần nhất và tự động ghép vào trước đoạn nói để STT không bị mất từ đầu câu.

2. **Sentence Boundary Chunking trong `LLMEngine`:**
   Thay vì chờ LLM sinh toàn bộ câu trả lời (mất 2-4 giây), engine phát hiện ngay dấu ngắt câu (`.`, `,`, `?`, `!`, `\n`) để gửi từng vế câu sang TTS. Nhờ đó, người dùng nghe thấy giọng nói phản hồi trong **dưới 1 giây** kể từ khi dứt lời.

3. **Cơ chế Ngắt Lời Tức Thì (Instant Barge-In):**
   Khi AI đang phát âm qua loa, nếu micro ghi nhận người dùng bắt đầu nói lại (`speech_start`), orchestrator lập tức gọi `audio_player.interrupt()`, giải phóng toàn bộ audio queue trong chưa đầy 10ms và hủy tác vụ sinh câu tiếp theo.
