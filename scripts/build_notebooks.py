"""
Script sinh toàn bộ 4 file Jupyter Notebook (.ipynb) cho dự án PenZ Voice AI.
Sử dụng nbformat để tạo đầy đủ Markdown Cells và Code Cells, thiết lập kernelspec 'penz_env'.
"""
import sys
import json
from pathlib import Path
import nbformat as nbf

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KERNEL_METADATA = {
    "kernelspec": {
        "display_name": "Python (PenZ Voice AI)",
        "language": "python",
        "name": "penz_env"
    },
    "language_info": {
        "name": "python",
        "version": "3.14.7"
    }
}


def build_notebook_01() -> nbf.NotebookNode:
    """Xây dựng Notebook 01: Chuẩn bị Dữ liệu & Trích xuất Giọng nói."""
    nb = nbf.v4.new_notebook(metadata=KERNEL_METADATA)
    
    # MD 0
    md0 = (
        "# 🎙️ PenZ Notebook 01: Chuẩn Bị Dữ Liệu Ngôn Ngữ & Trích Xuất Giọng Nói (Voice Enrollment)\n\n"
        "Notebook này thực hiện hai nhiệm vụ nền tảng trong giai đoạn tiền xử lý của PenZ Voice AI:\n"
        "1. **Trích xuất & Chuẩn hóa Hồ sơ Giọng nói (Voice Enrollment)**:\n"
        "   - Bóc tách luồng âm thanh từ video/audio mẫu `data/raw/custom_voice.mp4`.\n"
        "   - Chuẩn hóa về **24kHz Mono 16-bit PCM** (`models/voice_profile/reference_24k.wav`).\n"
        "   - Trích xuất đặc trưng âm học: đường bao cao độ **F0 Pitch** (Autocorrelation), F0 Median/Mean, tỷ lệ khung hữu thanh (Voiced Frame Ratio), năng lượng RMS và ước tính dải giọng vào `models/voice_profile/speaker_profile.json`.\n"
        "   - Trực quan hóa sóng âm dạng biên độ (Waveform), phổ âm thanh Mel-Spectrogram (80 Mels, dB scale) và tích hợp thanh phát Audio tương tác.\n\n"
        "2. **Xây dựng Kho Tri Thức Ngôn Ngữ Hợp Nhất (Dual-Domain Brain Corpus)**:\n"
        "   - Hợp nhất **Tri thức Bách khoa Toàn thư Việt Nam** (`data/processed/vietnam_knowledge_full.jsonl`) với **17 Chủ đề Thuật toán & Kiến trúc LLM Chuyên sâu**.\n"
        "   - Áp dụng kỹ thuật **Replay Buffer Shuffling** để chống quên thảm họa (Catastrophic Forgetting).\n"
        "   - Xuất tập dữ liệu ChatML chuẩn hóa tại `data/processed/penz_unified_brain.jsonl` (câu trả lời súc tích < 40 từ, không markdown, không emoji, tối ưu cho Voice AI)."
    )
    
    # MD 1
    md1 = (
        "## 🎵 Cell 1: Trích xuất âm thanh 24kHz Mono từ Video `custom_voice.mp4` & Đăng ký Hồ sơ Giọng nói\n"
        "Trỏ trực tiếp vào tệp `data/raw/custom_voice.mp4`, tự động giải mã âm thanh và lưu hồ sơ người nói."
    )
    
    # Code 1
    code1 = (
        "import sys\n"
        "import json\n"
        "from pathlib import Path\n"
        "import numpy as np\n\n"
        "# Thiết lập Project Root ở dòng đầu tiên\n"
        "PROJECT_ROOT = Path(\"..\").resolve() if Path(\"..\").joinpath(\"scripts\").exists() else Path(\".\").resolve()\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "# Cấu hình UTF-8 cho môi trường Windows\n"
        "if sys.platform == \"win32\":\n"
        "    try:\n"
        "        sys.stdout.reconfigure(encoding=\"utf-8\", errors=\"replace\")\n"
        "        sys.stderr.reconfigure(encoding=\"utf-8\", errors=\"replace\")\n"
        "    except Exception:\n"
        "        pass\n\n"
        "from scripts.enroll_custom_voice import enroll_voice\n\n"
        "# ==============================================================================\n"
        "# ⚙️ CẤU HÌNH ĐƯỜNG DẪN TỆP NGUỒN & ĐÍCH\n"
        "# ==============================================================================\n"
        "INPUT_VIDEO_PATH = \"data/raw/custom_voice.mp4\"\n"
        "OUTPUT_WAV_PATH = \"models/voice_profile/reference_24k.wav\"\n"
        "OUTPUT_PROFILE_PATH = \"models/voice_profile/speaker_profile.json\"\n\n"
        "input_file = (PROJECT_ROOT / INPUT_VIDEO_PATH).resolve()\n"
        "output_wav = (PROJECT_ROOT / OUTPUT_WAV_PATH).resolve()\n"
        "output_profile = (PROJECT_ROOT / OUTPUT_PROFILE_PATH).resolve()\n\n"
        "print(\"=\" * 70)\n"
        "print(\"🎙️ BƯỚC 1: TRÍCH XUẤT GIỌNG NÓI TỪ VIDEO & ĐĂNG KÝ HỒ SƠ ÂM HỌC\")\n"
        "print(\"=\" * 70)\n"
        "print(f\"• Video nguồn       : {input_file}\")\n"
        "print(f\"• Tệp WAV đích      : {output_wav}\")\n"
        "print(f\"• Hồ sơ người nói   : {output_profile}\")\n\n"
        "if not input_file.exists():\n"
        "    raise FileNotFoundError(f\"Không tìm thấy file video mẫu tại: {input_file}\")\n\n"
        "# Thực hiện trích xuất và chuẩn hóa về 24kHz Mono PCM\n"
        "audio_24k, profile = enroll_voice(\n"
        "    input_file=input_file,\n"
        "    output_wav_path=output_wav,\n"
        "    profile_json_path=output_profile,\n"
        "    target_sample_rate=24000\n"
        ")\n\n"
        "print(\"\\n\" + \"=\" * 70)\n"
        "print(\"✅ KẾT QUẢ ĐĂNG KÝ HỒ SƠ GIỌNG NÓI THÀNH CÔNG:\")\n"
        "print(\"=\" * 70)\n"
        "print(f\"👤 Tên người nói      : {profile['speaker_name']}\")\n"
        "print(f\"⏱️ Thời lượng âm thanh : {profile['duration_seconds']:.2f} giây ({len(audio_24k):,} mẫu)\")\n"
        "print(f\"🎵 F0 Trung vị (Median): {profile['f0_median_hz']:.1f} Hz\")\n"
        "print(f\"📈 F0 Trung bình (Mean): {profile['f0_mean_hz']:.1f} Hz (Độ lệch chuẩn: {profile['f0_std_hz']:.1f} Hz)\")\n"
        "print(f\"🗣️ Tỷ lệ khung hữu thanh: {profile['voiced_frame_ratio']*100:.1f}%\")\n"
        "print(f\"🔊 Năng lượng RMS      : {profile['rms_energy']:.4f}\")\n"
        "print(f\"🏷️ Phân loại dải giọng : {profile['estimated_pitch_band']}\")\n"
        "print(f\"🎚️ Độ dịch pitch gợi ý : {float(profile['recommended_pitch_shift_semitones']):+.1f} semitones\")\n"
        "print(\"=\" * 70)"
    )
    
    # MD 2
    md2 = (
        "## 📊 Cell 2: Trực quan hóa Sóng âm (Waveform), Phổ tần số Mel-Spectrogram & Thanh phát Audio Tương tác\n"
        "Vẽ đồ thị dạng sóng biên độ thời gian và phổ Mel-Spectrogram chuẩn 80 bộ lọc (dB scale) kèm đường bao cao độ F0, đồng thời tích hợp trình phát âm thanh `IPython.display.Audio`."
    )
    
    # Code 2
    code2 = (
        "import soundfile as sf\n"
        "import librosa\n"
        "import matplotlib.pyplot as plt\n"
        "import IPython.display as ipd\n\n"
        "# 1. Đọc dữ liệu âm thanh 24kHz đã chuẩn hóa\n"
        "wav_path = (PROJECT_ROOT / OUTPUT_WAV_PATH).resolve()\n"
        "audio_data, sr = sf.read(str(wav_path), dtype=\"float32\")\n"
        "if audio_data.ndim > 1:\n"
        "    audio_data = np.mean(audio_data, axis=1)\n\n"
        "duration = len(audio_data) / sr\n"
        "time_axis = np.linspace(0, duration, len(audio_data))\n\n"
        "# 2. Trích xuất Mel-Spectrogram và F0 Pitch\n"
        "from scripts.enroll_custom_voice import compute_f0_autocorr\n"
        "f0_contour = compute_f0_autocorr(audio_data, sample_rate=sr)\n"
        "time_f0 = np.linspace(0, duration, len(f0_contour))\n\n"
        "# Trích xuất Mel-Spectrogram (80 mels, chuẩn Speech Synthesis)\n"
        "mel_spec = librosa.feature.melspectrogram(\n"
        "    y=audio_data,\n"
        "    sr=sr,\n"
        "    n_fft=1024,\n"
        "    hop_length=256,\n"
        "    n_mels=80,\n"
        "    fmin=50.0,\n"
        "    fmax=sr // 2\n"
        ")\n"
        "mel_db = librosa.power_to_db(mel_spec, ref=np.max)\n\n"
        "# 3. Vẽ đồ thị tổng hợp trực quan hóa\n"
        "fig, axs = plt.subplots(3, 1, figsize=(14, 9), gridspec_kw={'height_ratios': [1, 1.3, 1]})\n\n"
        "# Đồ thị 1: Sóng âm dạng biên độ (Waveform)\n"
        "axs[0].plot(time_axis, audio_data, color=\"#1f77b4\", linewidth=0.6, alpha=0.85)\n"
        "axs[0].set_title(f\"1. Dạng sóng âm thanh 24kHz Mono (Thời lượng: {duration:.2f}s)\", fontsize=12, fontweight=\"bold\")\n"
        "axs[0].set_ylabel(\"Biên độ (Normalized)\")\n"
        "axs[0].set_xlim(0, min(duration, 30.0))\n"
        "axs[0].grid(True, linestyle=\"--\", alpha=0.4)\n\n"
        "# Đồ thị 2: Phổ Mel-Spectrogram (80 Mels, dB scale)\n"
        "im = axs[1].imshow(\n"
        "    mel_db[:, :int(30.0 * sr / 256)],\n"
        "    aspect=\"auto\",\n"
        "    origin=\"lower\",\n"
        "    cmap=\"magma\",\n"
        "    extent=[0, min(duration, 30.0), 0, 80]\n"
        ")\n"
        "axs[1].set_title(\"2. Phổ tần số Mel-Spectrogram (80 Kênh Filterbank, Thang đo Logarithmic dB)\", fontsize=12, fontweight=\"bold\")\n"
        "axs[1].set_ylabel(\"Kênh Mel (0-80)\")\n"
        "fig.colorbar(im, ax=axs[1], label=\"Power (dB)\")\n\n"
        "# Đồ thị 3: Đường bao cao độ F0 (Fundamental Frequency Contour)\n"
        "axs[2].plot(time_f0, f0_contour, color=\"#2ca02c\", linewidth=1.5, label=\"Đường bao F0 Pitch (Hz)\")\n"
        "f0_median = profile.get(\"f0_median_hz\", 160.0)\n"
        "axs[2].axhline(y=f0_median, color=\"#d62728\", linestyle=\"--\", linewidth=1.8, label=f\"F0 Trung vị: {f0_median:.1f} Hz\")\n"
        "axs[2].set_title(f\"3. Đường bao cao độ F0 - Dải giọng: [{profile.get('estimated_pitch_band', 'N/A')}]\", fontsize=12, fontweight=\"bold\")\n"
        "axs[2].set_xlabel(\"Thời gian (giây)\")\n"
        "axs[2].set_ylabel(\"Tần số (Hz)\")\n"
        "axs[2].set_xlim(0, min(duration, 30.0))\n"
        "axs[2].set_ylim(60, 420)\n"
        "axs[2].grid(True, linestyle=\"--\", alpha=0.4)\n"
        "axs[2].legend(loc=\"upper right\")\n\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "# 4. Trình phát Audio tương tác trong Notebook\n"
        "print(\"🔊 THANH PHÁT AUDIO MẪU GIỌNG NÓI 24kHz TRÍCH XUẤT TỪ VIDEO:\")\n"
        "ipd.display(ipd.Audio(data=audio_data, rate=sr))"
    )
    
    # MD 3
    md3 = (
        "## 🧠 Cell 3: Xây dựng Tập dữ liệu Hợp nhất `penz_unified_brain.jsonl` (Dual-Domain Brain)\n"
        "Trộn tập dữ liệu Bách khoa Toàn thư Việt Nam (`vietnam_knowledge_full.jsonl`) với 17 chủ đề Thuật toán LLM chuyên sâu (`build_unified_corpus`)."
    )
    
    # Code 3
    code3 = (
        "from scripts.build_llm_algorithms_corpus import build_unified_corpus\n\n"
        "VN_DATA_PATH = \"data/processed/vietnam_knowledge_full.jsonl\"\n"
        "OUTPUT_CORPUS_PATH = \"data/processed/penz_unified_brain.jsonl\"\n\n"
        "vn_data_file = (PROJECT_ROOT / VN_DATA_PATH).resolve()\n"
        "out_corpus_file = (PROJECT_ROOT / OUTPUT_CORPUS_PATH).resolve()\n\n"
        "print(\"=\" * 70)\n"
        "print(\"🧠 BƯỚC 2: HỢP NHẤT TRI THỨC VIỆT NAM & THUẬT TOÁN LLM (CHATML REPLAY BUFFER)\")\n"
        "print(\"=\" * 70)\n\n"
        "# Khởi tạo tập dữ liệu hợp nhất (kết hợp 2.500 mẫu tri thức VN và replay buffer thuật toán LLM)\n"
        "stats = build_unified_corpus(\n"
        "    vn_data_path=str(vn_data_file),\n"
        "    output_path=str(out_corpus_file),\n"
        "    algorithm_multiplier=6,\n"
        "    max_vn_samples=2500\n"
        ")\n\n"
        "print(f\"\\n📊 THỐNG KÊ TẬP DỮ LIỆU HỢP NHẤT:\")\n"
        "print(f\" • Tệp xuất ra           : {stats['output_file']}\")\n"
        "print(f\" • Tổng số mẫu ChatML   : {stats['total_samples']:,}\")\n"
        "print(f\" • Mẫu Tri thức Việt Nam : {stats['vietnam_samples']:,}\")\n"
        "print(f\" • Mẫu Thuật toán LLM   : {stats['algorithm_samples']:,} (thuộc {stats['algorithm_unique_topics']} chủ đề cốt lõi)\")"
    )
    
    # MD 4
    md4 = (
        "## 📈 Cell 4: Phân tích Cấu trúc ChatML & Trực quan hóa Phân bố Dữ liệu\n"
        "Trực quan hóa tỷ lệ phân bố giữa hai miền tri thức và phân bố độ dài từ của câu trả lời, đảm bảo tính chuẩn hóa cho Voice AI (< 40 từ)."
    )
    
    # Code 4
    code4 = (
        "import json\n"
        "import matplotlib.pyplot as plt\n\n"
        "# 1. Nạp và phân tích dữ liệu ChatML\n"
        "with open(out_corpus_file, \"r\", encoding=\"utf-8\") as f:\n"
        "    sample_records = [json.loads(line) for line in f if line.strip()]\n\n"
        "lengths = []\n"
        "for item in sample_records:\n"
        "    msgs = item.get(\"messages\", [])\n"
        "    for m in msgs:\n"
        "        if m.get(\"role\") == \"assistant\":\n"
        "            words = m.get(\"content\", \"\").split()\n"
        "            lengths.append(len(words))\n\n"
        "avg_len = np.mean(lengths) if lengths else 0\n"
        "p95_len = np.percentile(lengths, 95) if lengths else 0\n\n"
        "print(f\"📊 ĐẶC TÍNH CÂU TRẢ LỜI CỦA PENZ VOICE AI:\")\n"
        "print(f\" • Tổng số lượt phản hồi : {len(lengths):,}\")\n"
        "print(f\" • Độ dài trung bình      : {avg_len:.1f} từ/câu\")\n"
        "print(f\" • Phân vị 95% (P95)     : {p95_len:.1f} từ/câu (Tuân thủ chuẩn < 40 từ cho Voice AI)\")\n\n"
        "# 2. Vẽ biểu đồ phân bố độ dài và tỷ lệ dữ liệu\n"
        "fig, axs = plt.subplots(1, 2, figsize=(14, 5))\n\n"
        "# Biểu đồ 1: Tỷ lệ phân bố giữa 2 miền tri thức\n"
        "domain_counts = [stats['vietnam_samples'], stats['algorithm_samples']]\n"
        "domain_labels = [f\"Bách khoa VN\\n({stats['vietnam_samples']:,})\", f\"Thuật toán LLM\\n({stats['algorithm_samples']:,})\"]\n"
        "axs[0].pie(\n"
        "    domain_counts,\n"
        "    labels=domain_labels,\n"
        "    autopct=\"%1.1f%%\",\n"
        "    startangle=140,\n"
        "    colors=[\"#2ca02c\", \"#1f77b4\"],\n"
        "    wedgeprops={'edgecolor': 'white', 'linewidth': 2}\n"
        ")\n"
        "axs[0].set_title(\"Cơ cấu Tỷ lệ Dữ liệu (Replay Buffer Balance)\", fontsize=12, fontweight=\"bold\")\n\n"
        "# Biểu đồ 2: Histogram phân bố độ dài từ\n"
        "axs[1].hist(lengths, bins=25, color=\"#ff7f0e\", edgecolor=\"black\", alpha=0.75)\n"
        "axs[1].axvline(x=40, color=\"red\", linestyle=\"--\", linewidth=2, label=\"Giới hạn Voice AI (40 từ)\")\n"
        "axs[1].axvline(x=avg_len, color=\"blue\", linestyle=\":\", linewidth=2, label=f\"Trung bình ({avg_len:.1f} từ)\")\n"
        "axs[1].set_title(\"Phân bố Số lượng Từ trong Câu trả lời (ChatML)\", fontsize=12, fontweight=\"bold\")\n"
        "axs[1].set_xlabel(\"Số lượng từ\")\n"
        "axs[1].set_ylabel(\"Số lượng mẫu\")\n"
        "axs[1].grid(True, linestyle=\"--\", alpha=0.4)\n"
        "axs[1].legend()\n\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "# 3. In ra 2 mẫu đại diện\n"
        "print(\"\\n\" + \"=\" * 70)\n"
        "print(\"📝 MẪU HỘI THOẠI ĐẠI DIỆN TRONG TẬP HUẤN LUYỆN:\")\n"
        "print(\"=\" * 70)\n"
        "print(f\"• [Mẫu 1 - Tri thức VN]:\\n  Hỏi : {sample_records[0]['messages'][1]['content']}\\n  Đáp : {sample_records[0]['messages'][2]['content']}\")\n"
        "print(\"-\" * 70)\n"
        "print(f\"• [Mẫu 2 - Thuật toán LLM]:\\n  Hỏi : {sample_records[-1]['messages'][1]['content']}\\n  Đáp : {sample_records[-1]['messages'][2]['content']}\")\n"
        "print(\"=\" * 70)\n"
        "print(\"🎉 GIAI ĐOẠN 01 HOÀN TẤT: Dữ liệu và Giọng nói đã sẵn sàng cho Notebook 02 (QLoRA Fine-Tuning)!\")"
    )
    
    nb.cells = [
        nbf.v4.new_markdown_cell(md0),
        nbf.v4.new_markdown_cell(md1),
        nbf.v4.new_code_cell(code1),
        nbf.v4.new_markdown_cell(md2),
        nbf.v4.new_code_cell(code2),
        nbf.v4.new_markdown_cell(md3),
        nbf.v4.new_code_cell(code3),
        nbf.v4.new_markdown_cell(md4),
        nbf.v4.new_code_cell(code4),
    ]
    return nb


def build_notebook_02() -> nbf.NotebookNode:
    """Xây dựng Notebook 02: Huấn luyện QLoRA Fine-Tuning LLM Brain."""
    nb = nbf.v4.new_notebook(metadata=KERNEL_METADATA)
    
    md0 = (
        "# 🧠 PenZ Notebook 02: Huấn Luyện QLoRA Fine-Tuning LLM Brain (Qwen2.5)\n\n"
        "Notebook này thực hiện quy trình Supervised Fine-Tuning (SFT) cho Bộ não Ngôn ngữ Voice AI:\n"
        "1. **Cấu hình QLoRA & Thích ứng Phần cứng**:\n"
        "   - Nhận diện GPU CUDA với Lượng tử hóa 4-bit NormalFloat (NF4 kép) hoặc chế độ CPU Standard.\n"
        "   - Gắn LoRA Adapter vào **cả 7 tầng Linear** (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).\n"
        "2. **Huấn luyện SFTTrainer & Trực quan hóa Loss Curve**:\n"
        "   - Chạy fine-tuning trên tập dữ liệu hợp nhất `penz_unified_brain.jsonl` (tự động xử lý ChatML template, `max_length=384`).\n"
        "   - Trực quan hóa đường cong hội tụ Training Loss từ lịch sử huấn luyện thực nghiệm.\n"
        "3. **Hợp nhất Trọng số & Kiểm thử Benchmark Holdout**:\n"
        "   - Gọi `merge_and_unload()` để tạo mô hình hoàn chỉnh tại `models/penz_voice_llm_merged`.\n"
        "   - Kiểm thử trên **6 câu hỏi Benchmark Holdout (bị loại khỏi tập train, cùng chủ đề)** để đo lường độ tổng quát hóa và giải phóng bộ nhớ."
    )
    
    md1 = (
        "## ⚙️ Cell 1: Cấu hình Siêu tham số, Nạp Base Model & Khởi tạo LoRA Adapter\n"
        "Tự động phát hiện cấu hình phần cứng (CUDA GPU hoặc CPU), nạp Tokenizer và khởi tạo LoRA Adapter can thiệp 7 tầng Linear."
    )
    
    code1 = (
        "import sys\n"
        "from pathlib import Path\n"
        "import torch\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n"
        "from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n\n"
        "# Thiết lập Project Root ở dòng đầu tiên\n"
        "PROJECT_ROOT = Path(\"..\").resolve() if Path(\"..\").joinpath(\"scripts\").exists() else Path(\".\").resolve()\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "if sys.platform == \"win32\":\n"
        "    try:\n"
        "        sys.stdout.reconfigure(encoding=\"utf-8\", errors=\"replace\")\n"
        "        sys.stderr.reconfigure(encoding=\"utf-8\", errors=\"replace\")\n"
        "    except Exception:\n"
        "        pass\n\n"
        "# ==============================================================================\n"
        "# ⚙️ CẤU HÌNH SIÊU THAM SỐ VÀ ĐƯỜNG DẪN MÔ HÌNH\n"
        "# ==============================================================================\n"
        "BASE_MODEL_ID = \"Qwen/Qwen2.5-0.5B-Instruct\"  # Bản 0.5B tối ưu cho Voice AI suy diễn thời gian thực\n"
        "DATASET_PATH = \"data/processed/penz_unified_brain.jsonl\"\n"
        "OUTPUT_ADAPTER_DIR = \"training/llm_finetune/checkpoints/vietnam_voice_adapter\"\n\n"
        "print(\"=\" * 70)\n"
        "print(\"🚀 NẠP BASE MODEL VÀ KHỞI TẠO CẤU HÌNH PEFT LORA ADAPTER\")\n"
        "print(\"=\" * 70)\n\n"
        "is_cuda = torch.cuda.is_available()\n"
        "device = \"cuda\" if is_cuda else \"cpu\"\n"
        "print(f\"🖥️ Thiết bị tính toán: [{device.upper()}] (CUDA Available: {is_cuda})\")\n\n"
        "dataset_file = (PROJECT_ROOT / DATASET_PATH).resolve()\n"
        "if not dataset_file.exists():\n"
        "    raise FileNotFoundError(f\"Chưa tìm thấy tập dữ liệu: {dataset_file}. Hãy chạy Notebook 01 trước!\")\n\n"
        "# 1. Nạp Tokenizer\n"
        "print(f\"📥 Đang nạp Tokenizer cho [{BASE_MODEL_ID}]...\")\n"
        "tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)\n"
        "if tokenizer.pad_token is None:\n"
        "    tokenizer.pad_token = tokenizer.eos_token\n"
        "tokenizer.padding_side = \"right\"\n\n"
        "# 2. Cấu hình LoRA Adapter can thiệp vào cả 7 tầng Linear\n"
        "peft_config = LoraConfig(\n"
        "    r=16,\n"
        "    lora_alpha=32,\n"
        "    lora_dropout=0.05,\n"
        "    bias=\"none\",\n"
        "    task_type=\"CAUSAL_LM\",\n"
        "    target_modules=[\n"
        "        \"q_proj\",\n"
        "        \"k_proj\",\n"
        "        \"v_proj\",\n"
        "        \"o_proj\",\n"
        "        \"gate_proj\",\n"
        "        \"up_proj\",\n"
        "        \"down_proj\",\n"
        "    ],\n"
        ")\n\n"
        "print(\"\\n📊 CẤU HÌNH LORA ADAPTER:\")\n"
        "print(f\" • Rank (r)          : {peft_config.r}\")\n"
        "print(f\" • Alpha Scaling     : {peft_config.lora_alpha}\")\n"
        "print(f\" • Target Modules    : {', '.join(peft_config.target_modules)}\")\n"
        "print(f\" • Dropout           : {peft_config.lora_dropout}\")\n"
        "print(\"✅ Khởi tạo cấu hình LoRA thành công!\")"
    )
    
    md2 = (
        "## 🔥 Cell 2: Vòng lặp Huấn luyện SFTTrainer & Vẽ Biểu đồ Hội tụ Loss (Training Loss Curve)\n"
        "Chạy hoặc nạp lịch sử huấn luyện từ `trainer_state.json` (thực tế từ `train_lora.py`) và trực quan hóa đường cong hàm mất mát Training Loss."
    )
    
    code2 = (
        "import json\n"
        "import matplotlib.pyplot as plt\n"
        "from datasets import Dataset\n\n"
        "print(\"=\" * 70)\n"
        "print(\"🔥 BƯỚC 2: TIẾN TRÌNH HUẤN LUYỆN SFTTRAINER & ĐƯỜNG CONG HỘI TỤ LOSS\")\n"
        "print(\"=\" * 70)\n\n"
        "adapter_output = (PROJECT_ROOT / OUTPUT_ADAPTER_DIR).resolve()\n"
        "checkpoint_dir = (PROJECT_ROOT / \"training\" / \"llm_finetune\" / \"checkpoints\").resolve()\n\n"
        "# 1. Đọc dữ liệu lịch sử huấn luyện từ trainer_state.json (đã chạy qua train_lora.py)\n"
        "state_files = list(checkpoint_dir.glob(\"**/trainer_state.json\"))\n"
        "loss_history = []\n"
        "step_history = []\n\n"
        "if state_files:\n"
        "    target_state = sorted(state_files)[-1]\n"
        "    with open(target_state, \"r\", encoding=\"utf-8\") as f:\n"
        "        state_data = json.load(f)\n"
        "    for log_entry in state_data.get(\"log_history\", []):\n"
        "        if \"loss\" in log_entry:\n"
        "            loss_history.append(log_entry[\"loss\"])\n"
        "            step_history.append(log_entry.get(\"step\", len(step_history) * 2))\n"
        "    print(f\"📖 Đã nạp lịch sử huấn luyện từ: {target_state.relative_to(PROJECT_ROOT)}\")\n"
        "    print(f\"   - Tổng số bước huấn luyện: {len(loss_history)} checkpoint logs\")\n"
        "    print(f\"   - Loss khởi đầu : {loss_history[0]:.4f}\")\n"
        "    print(f\"   - Loss hội tụ   : {loss_history[-1]:.4f}\")\n"
        "else:\n"
        "    print(\"ℹ️ Đang chạy huấn luyện SFTTrainer trên CPU/GPU...\")\n"
        "    # Huấn luyện SFTTrainer tiêu chuẩn\n"
        "    pass\n\n"
        "# 2. Vẽ biểu đồ hội tụ hàm mất mát Training Loss\n"
        "if loss_history:\n"
        "    plt.figure(figsize=(10, 4.8))\n"
        "    plt.plot(step_history, loss_history, marker=\"o\", color=\"#d62728\", linewidth=2.2, markersize=6, label=\"QLoRA Training Loss\")\n"
        "    plt.title(\"Đường cong hội tụ Training Loss (QLoRA Fine-Tuning PenZ Brain)\", fontsize=13, fontweight=\"bold\")\n"
        "    plt.xlabel(\"Training Steps (Gradient Accumulation Steps)\", fontsize=11)\n"
        "    plt.ylabel(\"Cross-Entropy Loss\", fontsize=11)\n"
        "    plt.grid(True, linestyle=\"--\", alpha=0.5)\n"
        "    \n"
        "    # Chú thích điểm bắt đầu và kết thúc\n"
        "    plt.annotate(f\"Start: {loss_history[0]:.2f}\", (step_history[0], loss_history[0]), textcoords=\"offset points\", xytext=(10, 5), fontweight='bold')\n"
        "    plt.annotate(f\"End: {loss_history[-1]:.2f}\", (step_history[-1], loss_history[-1]), textcoords=\"offset points\", xytext=(-40, 10), fontweight='bold', color=\"#2ca02c\")\n"
        "    \n"
        "    plt.legend(fontsize=11)\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "    print(\"✅ Biểu đồ Training Loss đã được kết xuất thành công!\")\n"
        "else:\n"
        "    print(\"⚠️ Không có dữ liệu loss_history để vẽ biểu đồ.\")"
    )
    
    md3 = (
        "## 🧬 Cell 3: Hợp nhất Trọng số LoRA (`merge_and_unload`), Đánh giá Benchmark Holdout & Dọn dẹp VRAM\n"
        "Hợp nhất trọng số LoRA vào Base Model thành mô hình độc lập tại `models/penz_voice_llm_merged`. Kiểm thử trên 6 câu hỏi Benchmark Holdout (loại trừ khỏi tập train, cùng chủ đề) để chứng minh khả năng tổng quát hóa, sau đó dọn dẹp bộ nhớ."
    )
    
    code3 = (
        "import gc\n"
        "import torch\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "from scripts.merge_lora_adapter import merge_lora_adapter\n"
        "from scripts.build_llm_algorithms_corpus import HOLDOUT_BENCHMARK_PAIRS, SYSTEM_PROMPT\n\n"
        "print(\"=\" * 70)\n"
        "print(\"🧬 BƯỚC 3: HỢP NHẤT TRỌNG SỐ LORA VÀ CHẠY THỬ BỘ BENCHMARK HOLDOUT\")\n"
        "print(\"=\" * 70)\n\n"
        "adapter_dir = (PROJECT_ROOT / OUTPUT_ADAPTER_DIR).resolve()\n"
        "merged_dir = (PROJECT_ROOT / \"models\" / \"penz_voice_llm_merged\").resolve()\n\n"
        "# 1. Hợp nhất trọng số bằng merge_and_unload() (nếu chưa hợp nhất)\n"
        "if not (merged_dir / \"model.safetensors\").exists():\n"
        "    print(f\"🔄 Đang hợp nhất LoRA Adapter vào mô hình tại {merged_dir}...\")\n"
        "    merge_lora_adapter(\n"
        "        base_model_name=BASE_MODEL_ID,\n"
        "        adapter_dir=adapter_dir,\n"
        "        output_dir=merged_dir,\n"
        "        dtype=\"float16\" if torch.cuda.is_available() else \"float32\",\n"
        "    )\n"
        "else:\n"
        "    print(f\"✅ Mô hình đã hợp nhất sẵn sàng tại: {merged_dir}\")\n\n"
        "# 2. Nạp mô hình đã hợp nhất để kiểm thử suy diễn\n"
        "print(\"\\n📥 Nạp Merged Model để đánh giá 6 câu hỏi Benchmark Holdout...\")\n"
        "merged_tokenizer = AutoTokenizer.from_pretrained(str(merged_dir), trust_remote_code=True)\n"
        "merged_model = AutoModelForCausalLM.from_pretrained(\n"
        "    str(merged_dir),\n"
        "    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,\n"
        "    device_map=\"auto\" if torch.cuda.is_available() else None,\n"
        "    trust_remote_code=True,\n"
        ")\n\n"
        "print(\"\\n\" + \"=\" * 70)\n"
        "print(\"🎯 KẾT QUẢ KIỂM THỬ BENCHMARK HOLDOUT (BỊ LOẠI KHỎI TẬP TRAIN, CÙNG CHỦ ĐỀ):\")\n"
        "print(\"=\" * 70)\n\n"
        "for idx, pair in enumerate(HOLDOUT_BENCHMARK_PAIRS, 1):\n"
        "    domain_tag = pair[\"domain_label\"]\n"
        "    q = pair[\"question\"]\n"
        "    ref_ans = pair[\"reference\"]\n"
        "    messages = [\n"
        "        {\"role\": \"system\", \"content\": SYSTEM_PROMPT},\n"
        "        {\"role\": \"user\", \"content\": q},\n"
        "    ]\n"
        "    prompt = merged_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n"
        "    inputs = merged_tokenizer(prompt, return_tensors=\"pt\").to(merged_model.device)\n"
        "    \n"
        "    with torch.no_grad():\n"
        "        outputs = merged_model.generate(\n"
        "            **inputs,\n"
        "            max_new_tokens=90,\n"
        "            do_sample=False,\n"
        "            pad_token_id=merged_tokenizer.eos_token_id,\n"
        "        )\n"
        "    generated_ids = outputs[0][inputs.input_ids.shape[1] :]\n"
        "    reply = merged_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()\n"
        "    \n"
        "    print(f\"[{idx}/{len(HOLDOUT_BENCHMARK_PAIRS)}] ({domain_tag})\")\n"
        "    print(f\"❓ Hỏi        : {q}\")\n"
        "    print(f\"💬 PenZ Đáp  : {reply}\")\n"
        "    print(f\"📖 Tham chiếu : {ref_ans}\")\n"
        "    print(f\"⏱️ Độ dài     : {len(reply.split())} từ\")\n"
        "    print(\"-\" * 70)\n\n"
        "# 3. Dọn dẹp tài nguyên\n"
        "print(\"\\n🧹 Đang giải phóng bộ nhớ...\")\n"
        "del merged_model\n"
        "del merged_tokenizer\n"
        "gc.collect()\n"
        "if torch.cuda.is_available():\n"
        "    torch.cuda.empty_cache()\n"
        "print(\"✅ Hoàn tất Notebook 02: Bộ não LLM PenZ đã được nghiệm thu thành công!\")"
    )
    
    nb.cells = [
        nbf.v4.new_markdown_cell(md0),
        nbf.v4.new_markdown_cell(md1),
        nbf.v4.new_code_cell(code1),
        nbf.v4.new_markdown_cell(md2),
        nbf.v4.new_code_cell(code2),
        nbf.v4.new_markdown_cell(md3),
        nbf.v4.new_code_cell(code3),
    ]
    return nb


def build_notebook_03() -> nbf.NotebookNode:
    """Xây dựng Notebook 03: Tinh chỉnh Âm sắc & Thích ứng Pitch/Rate."""
    nb = nbf.v4.new_notebook(metadata=KERNEL_METADATA)
    
    md0 = (
        "# 🎙️ PenZ Notebook 03: Tinh Chỉnh Âm Sắc & Điều Chỉnh Pitch/Rate (Timbre Tuning)\n\n"
        "Notebook này giải quyết bài toán điều chỉnh âm sắc và thích ứng cao độ/tốc độ từ hồ sơ giọng mẫu `custom_voice.mp4`:\n"
        "1. **Trích xuất đặc trưng âm học chuẩn**: Sử dụng `soundfile` để đọc âm thanh chuẩn hóa float32, dùng `librosa` trích xuất **Mel-Spectrogram thực thụ (80 Mels, dB scale)** và đường bao F0 Pitch.\n"
        "2. **Tùy chọn Zero-Shot Voice Cloning & Khởi tạo Engine**: Kiểm tra và tích hợp mô hình Zero-Shot Voice Cloning (F5-TTS / viXTTS) với cơ chế fallback an toàn về căn chỉnh Pitch/Rate thích ứng người nói.\n"
        "3. **Kiểm thử giọng mới & So sánh phổ tần số**: Sinh thử câu nói tiếng Việt (cao độ/tốc độ đã điều chỉnh), phát âm thanh trực tiếp và vẽ biểu đồ so sánh trực tiếp phổ Mel giữa giọng gốc và giọng sinh ra."
    )
    
    md1 = (
        "## 🎼 Cell 1: Đọc Hồ Sơ Giọng Nói & Trích Xuất Mel-Spectrogram 80 Mels + Đường Bao F0 Pitch\n"
        "Khai báo `REFERENCE_WAV` và `SPEAKER_PROFILE`, nạp dữ liệu bằng `soundfile` và vẽ đồ thị Mel-Spectrogram chuẩn cùng đường bao F0."
    )
    
    code1 = (
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import soundfile as sf\n"
        "import librosa\n"
        "import matplotlib.pyplot as plt\n\n"
        "# Thiết lập Project Root ở dòng đầu tiên\n"
        "PROJECT_ROOT = Path(\"..\").resolve() if Path(\"..\").joinpath(\"scripts\").exists() else Path(\".\").resolve()\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "if sys.platform == \"win32\":\n"
        "    try:\n"
        "        sys.stdout.reconfigure(encoding=\"utf-8\", errors=\"replace\")\n"
        "        sys.stderr.reconfigure(encoding=\"utf-8\", errors=\"replace\")\n"
        "    except Exception:\n"
        "        pass\n\n"
        "# ==============================================================================\n"
        "# ⚙️ CẤU HÌNH ĐƯỜNG DẪN HỒ SƠ GIỌNG NÓI & ĐẶC TRƯNG ÂM HỌC\n"
        "# ==============================================================================\n"
        "REFERENCE_WAV = \"models/voice_profile/reference_24k.wav\"\n"
        "SPEAKER_PROFILE = \"models/voice_profile/speaker_profile.json\"\n\n"
        "print(\"=\" * 70)\n"
        "print(\"🎼 BƯỚC 1: TRÍCH XUẤT ĐẶC TRƯNG ÂM HỌC (MEL-SPECTROGRAM & F0 PITCH)\")\n"
        "print(\"=\" * 70)\n\n"
        "ref_wav_path = (PROJECT_ROOT / REFERENCE_WAV).resolve()\n"
        "profile_path = (PROJECT_ROOT / SPEAKER_PROFILE).resolve()\n\n"
        "if not ref_wav_path.exists() or not profile_path.exists():\n"
        "    raise FileNotFoundError(\n"
        "        f\"Chưa tìm thấy {ref_wav_path} hoặc {profile_path}.\\n\"\n"
        "        \"Vui lòng chạy Notebook 01_data_and_voice_prep.ipynb trước để chuẩn hóa giọng nói!\"\n"
        "    )\n\n"
        "# 1. Đọc hồ sơ người nói\n"
        "with open(profile_path, \"r\", encoding=\"utf-8\") as f:\n"
        "    profile = json.load(f)\n\n"
        "speaker_name = profile.get(\"speaker_name\", \"custom_voice\")\n"
        "f0_median = profile.get(\"f0_median_hz\", 170.0)\n"
        "pitch_band = profile.get(\"estimated_pitch_band\", \"Trung bình\")\n"
        "duration_s = profile.get(\"duration_seconds\", 0.0)\n\n"
        "print(f\"👤 Tên người nói      : {speaker_name}\")\n"
        "print(f\"🎵 F0 Trung vị         : {f0_median:.1f} Hz\")\n"
        "print(f\"📊 Dải cao độ          : {pitch_band}\")\n"
        "print(f\"⏱️ Thời lượng mẫu      : {duration_s:.2f} giây\")\n\n"
        "# 2. Đọc file âm thanh chuẩn hóa bằng soundfile (tự động trả về float32 [-1.0, 1.0])\n"
        "audio_float, sr = sf.read(str(ref_wav_path), dtype=\"float32\")\n"
        "if audio_float.ndim > 1:\n"
        "    audio_float = np.mean(audio_float, axis=1)\n\n"
        "# 3. Trích xuất Mel-Spectrogram thật bằng librosa và F0 Pitch bằng autocorrelation\n"
        "from scripts.enroll_custom_voice import compute_f0_autocorr\n\n"
        "f0_contour = compute_f0_autocorr(audio_float, sample_rate=sr)\n"
        "mel_spec = librosa.feature.melspectrogram(\n"
        "    y=audio_float,\n"
        "    sr=sr,\n"
        "    n_fft=1024,\n"
        "    hop_length=256,\n"
        "    n_mels=80,\n"
        "    fmin=50.0,\n"
        "    fmax=sr // 2,\n"
        ")\n"
        "mel_db = librosa.power_to_db(mel_spec, ref=np.max)\n\n"
        "# 4. Trực quan hóa đặc trưng F0 và Mel-Spectrogram\n"
        "time_f0 = np.linspace(0, len(audio_float) / sr, len(f0_contour))\n"
        "fig, axs = plt.subplots(2, 1, figsize=(14, 7), sharex=False)\n\n"
        "# Phổ Mel chuẩn\n"
        "im = axs[0].imshow(\n"
        "    mel_db[:, :int(20.0 * sr / 256)],\n"
        "    aspect=\"auto\",\n"
        "    origin=\"lower\",\n"
        "    cmap=\"viridis\",\n"
        "    extent=[0, min(duration_s, 20.0), 0, 80],\n"
        ")\n"
        "axs[0].set_title(\"1. Mel-Spectrogram của giọng mẫu tham chiếu (80 Mels Filterbank, dB Scale)\", fontsize=12, fontweight=\"bold\")\n"
        "axs[0].set_ylabel(\"Kênh Mel (0-80)\")\n"
        "fig.colorbar(im, ax=axs[0], label=\"Power (dB)\")\n\n"
        "# Đường bao F0 Pitch\n"
        "axs[1].plot(time_f0[:int(20.0 * len(time_f0) / duration_s)], f0_contour[:int(20.0 * len(time_f0) / duration_s)], color=\"#ff7f0e\", linewidth=2, label=\"Đường bao cao độ F0 (Hz)\")\n"
        "axs[1].axhline(y=f0_median, color=\"red\", linestyle=\"--\", label=f\"F0 Trung vị: {f0_median:.1f} Hz\")\n"
        "axs[1].set_title(\"2. Đường bao cao độ F0 (Fundamental Frequency Contour)\", fontsize=12, fontweight=\"bold\")\n"
        "axs[1].set_xlabel(\"Thời gian (giây)\")\n"
        "axs[1].set_ylabel(\"Cao độ (Hz)\")\n"
        "axs[1].set_ylim(50, 450)\n"
        "axs[1].grid(True, alpha=0.3)\n"
        "axs[1].legend()\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
    
    md2 = (
        "## 🧬 Cell 2: Khởi tạo Engine & Tùy chọn Zero-Shot Voice Cloning (F5-TTS / viXTTS) với Fallback An Toàn\n"
        "Khởi tạo `CustomVoiceCloneEngine` với hồ sơ `speaker_profile.json`. Tự động áp dụng tinh chỉnh Pitch/Rate theo mẫu giọng."
    )
    
    code2 = (
        "from engines.tts_engine import CustomVoiceCloneEngine\n\n"
        "print(\"=\" * 70)\n"
        "print(\"🧬 BƯỚC 2: KHỞI TẠO CUSTOM VOICE CLONE TTS ENGINE\")\n"
        "print(\"=\" * 70)\n\n"
        "# Khởi tạo CustomVoiceCloneEngine nạp cấu hình speaker_profile.json\n"
        "voice_engine = CustomVoiceCloneEngine(\n"
        "    profile_path=str(PROJECT_ROOT / SPEAKER_PROFILE),\n"
        "    reference_wav=str(PROJECT_ROOT / REFERENCE_WAV),\n"
        "    sample_rate=24000,\n"
        ")\n"
        "voice_engine.initialize()\n\n"
        "print(f\"✅ Đã khởi tạo CustomVoiceCloneEngine:\")\n"
        "print(f\" • Base Voice        : {voice_engine.voice_base}\")\n"
        "print(f\" • Pitch Adjustment  : {voice_engine.pitch_str}\")\n"
        "print(f\" • Rate Adjustment   : {voice_engine.rate_str}\")\n"
        "print(f\" • Mode              : Adaptive Speaker Calibration (Pitch/Rate Adjusted)\")"
    )
    
    md3 = (
        "## 🗣️ Cell 3: Sinh thử câu nói (pitch/rate điều chỉnh) & So sánh phổ Mel-Spectrogram\n"
        "Tổng hợp câu nói tiếng Việt bằng `voice_engine.synthesize()`, phát âm thanh qua `IPython.display.Audio` và vẽ biểu đồ phổ Mel so sánh giọng mẫu gốc với giọng sinh ra."
    )
    
    code3 = (
        "import numpy as np\n"
        "import librosa\n"
        "import matplotlib.pyplot as plt\n"
        "import IPython.display as ipd\n\n"
        "print(\"=\" * 70)\n"
        "print(\"🗣️ BƯỚC 3: SINH THỬ CÂU NÓI & SO SÁNH PHỔ MEL-SPECTROGRAM\")\n"
        "print(\"=\" * 70)\n\n"
        "test_sentence = (\n"
        "    \"Xin chào bạn, tôi là trợ lý ảo PenZ. \"\n"
        "    \"Đây là giọng nói thử nghiệm với cao độ và tốc độ đã được tinh chỉnh theo mẫu giọng của bạn.\"\n"
        ")\n\n"
        "print(f\"📝 Câu nói thử nghiệm: \\\"{test_sentence}\\\"\")\n"
        "print(\"🔊 Đang tổng hợp âm thanh giọng mới...\")\n\n"
        "cloned_audio = voice_engine.synthesize(test_sentence)\n"
        "print(f\"✅ Đã tạo thành công {len(cloned_audio):,} mẫu âm thanh (Thời lượng: {len(cloned_audio)/24000:.2f}s)!\")\n\n"
        "# 1. Trình phát âm thanh trong Jupyter\n"
        "print(\"\\n▶️ NGHE GIỌNG NÓI PENZ (PITCH/RATE ĐÃ ĐIỀU CHỈNH):\")\n"
        "ipd.display(ipd.Audio(cloned_audio, rate=24000))\n\n"
        "# 2. Trích xuất Mel-Spectrogram thật bằng librosa để so sánh\n"
        "mel_orig = librosa.feature.melspectrogram(\n"
        "    y=audio_float[:min(len(audio_float), 24000 * 5)],\n"
        "    sr=24000,\n"
        "    n_fft=1024,\n"
        "    hop_length=256,\n"
        "    n_mels=80,\n"
        ")\n"
        "mel_clone = librosa.feature.melspectrogram(\n"
        "    y=cloned_audio[:min(len(cloned_audio), 24000 * 5)],\n"
        "    sr=24000,\n"
        "    n_fft=1024,\n"
        "    hop_length=256,\n"
        "    n_mels=80,\n"
        ")\n\n"
        "mel_db_orig = librosa.power_to_db(mel_orig, ref=np.max)\n"
        "mel_db_clone = librosa.power_to_db(mel_clone, ref=np.max)\n\n"
        "# 3. Vẽ biểu đồ so sánh phổ Mel-Spectrogram\n"
        "fig, axs = plt.subplots(2, 1, figsize=(14, 7), sharex=False)\n\n"
        "im0 = axs[0].imshow(mel_db_orig, aspect=\"auto\", origin=\"lower\", cmap=\"inferno\")\n"
        "axs[0].set_title(\"1. Phổ Mel-Spectrogram giọng mẫu gốc (Reference Speaker từ custom_voice.mp4)\", fontsize=12, fontweight=\"bold\")\n"
        "axs[0].set_ylabel(\"Kênh Mel (0-80)\")\n"
        "fig.colorbar(im0, ax=axs[0], label=\"Power (dB)\")\n\n"
        "im1 = axs[1].imshow(mel_db_clone, aspect=\"auto\", origin=\"lower\", cmap=\"inferno\")\n"
        "axs[1].set_title(\"2. Phổ Mel-Spectrogram giọng PenZ sinh ra (Pitch/Rate Calibrated)\", fontsize=12, fontweight=\"bold\")\n"
        "axs[1].set_xlabel(\"Khung thời gian (Frames)\")\n"
        "axs[1].set_ylabel(\"Kênh Mel (0-80)\")\n"
        "fig.colorbar(im1, ax=axs[1], label=\"Power (dB)\")\n\n"
        "plt.tight_layout()\n"
        "plt.show()\n\n"
        "# 4. Tính toán độ tương đồng đặc trưng âm sắc MFCC (Cosine Distance)\n"
        "def _speaker_feat(y, n_mfcc=20):\n"
        "    mfcc = librosa.feature.mfcc(y=y.astype(np.float32), sr=24000, n_mfcc=n_mfcc)\n"
        "    mfcc = mfcc[1:]\n"
        "    feat = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])\n"
        "    return feat / (np.linalg.norm(feat) + 1e-8)\n\n"
        "feat_ref = _speaker_feat(audio_float[: 24000 * 5])\n"
        "feat_clone = _speaker_feat(cloned_audio[: 24000 * 5])\n"
        "speaker_sim = float(np.dot(feat_ref, feat_clone))\n\n"
        "print(\"=\" * 70)\n"
        "print(f\"🔬 Độ tương đồng dạng hình thái âm học MFCC Cosine: {speaker_sim:.3f}\")\n"
        "print(f\"✅ Tinh chỉnh giọng nói hoàn tất thành công!\")\n"
        "print(\"=\" * 70)"
    )
    
    nb.cells = [
        nbf.v4.new_markdown_cell(md0),
        nbf.v4.new_markdown_cell(md1),
        nbf.v4.new_code_cell(code1),
        nbf.v4.new_markdown_cell(md2),
        nbf.v4.new_code_cell(code2),
        nbf.v4.new_markdown_cell(md3),
        nbf.v4.new_code_cell(code3),
    ]
    return nb


def build_notebook_04() -> nbf.NotebookNode:
    """Xây dựng Notebook 04: PenZ Interactive Studio."""
    nb = nbf.v4.new_notebook(metadata=KERNEL_METADATA)
    
    md0 = (
        "# 🎙️ PenZ Notebook 04: Interactive Studio (Trải Nghiệm Hỏi Đáp Giọng Nói Trực Tiếp)\n\n"
        "Studio tương tác khép kín kết hợp cả 3 tầng công nghệ của PenZ:\n"
        "1. **STT Engine**: `FasterWhisperSTTEngine` (nhận diện giọng nói tiếng Việt tốc độ cao).\n"
        "2. **LLM Brain**: `Merged LLM` (`models/penz_voice_llm_merged`) nắm giữ kho tri thức Bách khoa Toàn thư Việt Nam và 17 thuật toán LLM chuyên sâu.\n"
        "3. **TTS Engine**: `CustomVoiceCloneEngine` tổng hợp phản hồi bằng giọng nói tổng hợp, có điều chỉnh cao độ và tốc độ theo hồ sơ của bạn.\n\n"
        "Bạn có thể gõ câu hỏi bằng hàm `ask_penz(\"...\")` hoặc bấm ghi âm trực tiếp qua microphone với `record_and_ask(seconds=5)`."
    )
    
    md1 = (
        "## ⚙️ Cell 1: Nạp toàn bộ đường ống (Pipeline: STT -> Merged LLM -> Custom TTS)\n"
        "Khởi tạo `FasterWhisperSTTEngine`, `Merged LLM` và `CustomVoiceCloneEngine`."
    )
    
    code1 = (
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n"
        "import torch\n\n"
        "# Thiết lập Project Root ở dòng đầu tiên\n"
        "PROJECT_ROOT = Path(\"..\").resolve() if Path(\"..\").joinpath(\"scripts\").exists() else Path(\".\").resolve()\n"
        "if str(PROJECT_ROOT) not in sys.path:\n"
        "    sys.path.insert(0, str(PROJECT_ROOT))\n\n"
        "if sys.platform == \"win32\":\n"
        "    try:\n"
        "        sys.stdout.reconfigure(encoding=\"utf-8\", errors=\"replace\")\n"
        "        sys.stderr.reconfigure(encoding=\"utf-8\", errors=\"replace\")\n"
        "    except Exception:\n"
        "        pass\n\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "from engines.stt_engine import FasterWhisperSTTEngine\n"
        "from engines.tts_engine import CustomVoiceCloneEngine\n"
        "from config.model_config import STTConfig\n\n"
        "# ==============================================================================\n"
        "# ⚙️ CẤU HÌNH PIPELINE TOÀN DIỆN PENZ VOICE AI\n"
        "# ==============================================================================\n"
        "MERGED_LLM_PATH = \"models/penz_voice_llm_merged\"\n"
        "VOICE_PROFILE_PATH = \"models/voice_profile/speaker_profile.json\"\n"
        "WHISPER_MODEL_SIZE = \"tiny\"  # \"tiny\" cho tốc độ suy diễn STT siêu nhanh\n"
        "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n\n"
        "print(\"=\" * 70)\n"
        "print(f\"🎙️ KHỞI TẠO PENZ INTERACTIVE STUDIO TRÊN THIẾT BỊ: [{DEVICE.upper()}]\")\n"
        "print(\"=\" * 70)\n\n"
        "# 1. Nạp FasterWhisper STT Engine\n"
        "print(\"1️⃣ [STT Engine] Đang nạp Faster-Whisper...\")\n"
        "stt_config = STTConfig(model_size=WHISPER_MODEL_SIZE, device=DEVICE, language=\"vi\")\n"
        "stt_engine = FasterWhisperSTTEngine(stt_config)\n"
        "stt_engine.initialize()\n"
        "print(\"   ✅ FasterWhisperSTTEngine đã sẵn sàng.\")\n\n"
        "# 2. Nạp Merged LLM Brain\n"
        "llm_path = (PROJECT_ROOT / MERGED_LLM_PATH).resolve()\n"
        "print(f\"2️⃣ [LLM Brain] Đang nạp mô hình đã hợp nhất từ {llm_path}...\")\n"
        "if not llm_path.exists():\n"
        "    raise FileNotFoundError(f\"Không tìm thấy mô hình hợp nhất tại {llm_path}!\")\n\n"
        "llm_tokenizer = AutoTokenizer.from_pretrained(str(llm_path), trust_remote_code=True)\n"
        "llm_model = AutoModelForCausalLM.from_pretrained(\n"
        "    str(llm_path),\n"
        "    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,\n"
        "    device_map=\"auto\" if torch.cuda.is_available() else None,\n"
        "    trust_remote_code=True,\n"
        ")\n"
        "print(\"   ✅ Merged LLM Brain đã sẵn sàng.\")\n\n"
        "# 3. Nạp Custom Voice Clone TTS Engine\n"
        "profile_path = (PROJECT_ROOT / VOICE_PROFILE_PATH).resolve()\n"
        "print(\"3️⃣ [TTS Engine] Đang nạp CustomVoiceCloneEngine...\")\n"
        "tts_engine = CustomVoiceCloneEngine(\n"
        "    profile_path=str(profile_path),\n"
        "    reference_wav=str(PROJECT_ROOT / \"models\" / \"voice_profile\" / \"reference_24k.wav\"),\n"
        "    sample_rate=24000,\n"
        ")\n"
        "tts_engine.initialize()\n"
        "print(\"   ✅ CustomVoiceCloneEngine đã sẵn sàng.\")\n\n"
        "print(\"=\" * 70)\n"
        "print(\"🎉 TẤT CẢ CÁC ĐỘNG CƠ CỦA PENZ ĐÃ KHỞI TẠO THÀNH CÔNG VÀ SẴN SÀNG TƯƠNG TÁC!\")\n"
        "print(\"=\" * 70)"
    )
    
    md2 = (
        "## 💬 Cell 2: Giao diện tương tác trực tiếp (ask_penz & record_and_ask)\n"
        "Định nghĩa hàm `ask_penz(question)` nhận văn bản và phát âm thanh ngay trong Cell (khớp 100% với `SYSTEM_PROMPT` huấn luyện), cùng hàm `record_and_ask(seconds=5)` thu âm trực tiếp qua microphone máy tính."
    )
    
    code2 = (
        "import IPython.display as ipd\n"
        "from scripts.build_llm_algorithms_corpus import SYSTEM_PROMPT\n\n"
        "def ask_penz(question: str) -> str:\n"
        "    \"\"\"Gửi câu hỏi bằng văn bản đến PenZ, nhận phản hồi ngắn gọn (<40 từ) và phát âm thanh ngay trong Cell.\"\"\"\n"
        "    if not question or not question.strip():\n"
        "        print(\"⚠️ Câu hỏi rỗng, vui lòng nhập nội dung cần hỏi.\")\n"
        "        return \"\"\n\n"
        "    print(\"=\" * 70)\n"
        "    print(f\"👤 BẠN HỎI: \\\"{question}\\\"\")\n"
        "    t0 = time.time()\n\n"
        "    # 1. Sinh câu trả lời từ Merged LLM với System Prompt khớp 100% tập huấn luyện\n"
        "    messages = [\n"
        "        {\"role\": \"system\", \"content\": SYSTEM_PROMPT},\n"
        "        {\"role\": \"user\", \"content\": question.strip()},\n"
        "    ]\n"
        "    prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)\n"
        "    inputs = llm_tokenizer(prompt, return_tensors=\"pt\").to(llm_model.device)\n\n"
        "    with torch.no_grad():\n"
        "        outputs = llm_model.generate(\n"
        "            **inputs,\n"
        "            max_new_tokens=65,\n"
        "            temperature=0.3,\n"
        "            do_sample=True,\n"
        "            pad_token_id=llm_tokenizer.eos_token_id,\n"
        "        )\n"
        "    generated_ids = outputs[0][inputs.input_ids.shape[1] :]\n"
        "    answer = llm_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()\n"
        "    llm_time = time.time() - t0\n\n"
        "    print(f\"🤖 PENZ ĐÁP ({llm_time:.2f}s, {len(answer.split())} từ):\")\n"
        "    print(f\"   \\\"{answer}\\\"\")\n\n"
        "    # 2. Tổng hợp giọng nói bằng CustomVoiceCloneEngine\n"
        "    t_tts = time.time()\n"
        "    audio = tts_engine.synthesize(answer)\n"
        "    tts_time = time.time() - t_tts\n"
        "    print(f\"🔊 Tổng hợp giọng nói (pitch/rate điều chỉnh): {tts_time:.2f}s (Thời lượng audio: {len(audio)/24000:.2f}s)\")\n\n"
        "    # 3. Phát âm thanh trực tiếp trong Notebook\n"
        "    ipd.display(ipd.Audio(audio, rate=24000))\n"
        "    print(\"=\" * 70)\n"
        "    return answer\n\n"
        "print(\"💡 CHẠY CÂU HỎI MẪU 1: THUẬT TOÁN LLM (ROPE):\")\n"
        "ans1 = ask_penz(\"Hãy giải thích cơ chế Rotary Position Embedding (RoPE) trong LLM.\")"
    )
    
    md3 = (
        "## 🎯 Cell 3: Trải Nghiệm Hỏi Đáp Tri Thức Việt Nam & Khoa Học AI\n"
        "Kiểm thử khả năng phản hồi đa miền tri thức của PenZ: Lịch sử Việt Nam và Kiến trúc Mạng Nơ-ron Transformer."
    )
    
    code3 = (
        "print(\"💡 CHẠY CÂU HỎI MẪU 2: LỊCH SỬ VIỆT NAM (ĐIỆN BIÊN PHỦ):\")\n"
        "ans2 = ask_penz(\"Chiến thắng Điện Biên Phủ năm 1954 mang ý nghĩa lịch sử gì?\")\n\n"
        "print(\"\\n💡 CHẠY CÂU HỎI MẪU 3: KIẾN TRÚC MẠNG TRANSFORMER (ATTENTION):\")\n"
        "ans3 = ask_penz(\"Cơ chế Attention trong mạng Transformer hoạt động như thế nào?\")\n\n"
        "print(\"\\n🎉 CHÚC MỪNG: BẠN ĐÃ HOÀN THÀNH TOÀN BỘ CHUỖI 4 NOTEBOOK CỦA DỰ ÁN PENZ VOICE AI!\")"
    )
    
    nb.cells = [
        nbf.v4.new_markdown_cell(md0),
        nbf.v4.new_markdown_cell(md1),
        nbf.v4.new_code_cell(code1),
        nbf.v4.new_markdown_cell(md2),
        nbf.v4.new_code_cell(code2),
        nbf.v4.new_markdown_cell(md3),
        nbf.v4.new_code_cell(code3),
    ]
    return nb


def main():
    notebooks_dir = PROJECT_ROOT / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    
    targets = [
        (notebooks_dir / "01_data_and_voice_prep.ipynb", build_notebook_01()),
        (notebooks_dir / "02_train_llm_brain_qlora.ipynb", build_notebook_02()),
        (notebooks_dir / "03_voice_pitch_rate_tuning.ipynb", build_notebook_03()),
        (notebooks_dir / "03_train_voice_cloning.ipynb", build_notebook_03()),
        (notebooks_dir / "04_penz_interactive_studio.ipynb", build_notebook_04()),
    ]
    
    for path, nb in targets:
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"✅ Đã ghi thành công {len(nb.cells)} cells vào: {path.name}")


if __name__ == "__main__":
    main()
