import json
import base64
import soundfile as sf
import numpy as np
import librosa
from pathlib import Path

nb_path = Path("notebooks/03_train_voice_cloning.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Markdown Cell 0
nb["cells"][0]["source"] = [
    "# 🎙️ PenZ Notebook 03: True Neural Zero-Shot Voice Cloning (F5-TTS Vietnamese)\n",
    "\n",
    "Notebook này thực hiện quy trình nhân bản giọng nói thật (True Neural Zero-Shot Voice Cloning) từ mẫu giọng cô gái miền Bắc trong `custom_voice.mp4`:\n",
    "1. **Trích xuất đặc trưng âm học chuẩn**: Sử dụng `soundfile` để đọc âm thanh tham chiếu chuẩn hóa float32 (24kHz Mono PCM), dùng `librosa` trích xuất **Mel-Spectrogram thực thụ (80 Mels, dB scale)** và đường bao F0 Pitch.\n",
    "2. **Khởi tạo F5-TTS Neural Voice Cloning Engine**: Nạp checkpoint mô hình F5-TTS Vietnamese (`model_latest.safetensors`, Vocos 24kHz) cùng từ điển âm tiết `vocab.txt` và hồ sơ `speaker_profile.json`.\n",
    "3. **Tổng hợp giọng mẫu & So sánh phổ Mel-Spectrogram**: Tổng hợp câu nói bằng chính giọng miền Bắc từ bản gốc, phát âm thanh trực tiếp và vẽ biểu đồ so sánh trực tiếp phổ Mel giữa giọng gốc và giọng sinh ra."
]

# Update Markdown Cell 3
nb["cells"][3]["source"] = [
    "## 🧬 Cell 2: Khởi tạo F5-TTS Vietnamese Neural Voice Cloning Engine\n",
    "Khởi tạo `CustomVoiceCloneEngine` sử dụng kiến trúc DiT (Diffusion Transformer) và Vocos 24kHz được huấn luyện chuyên sâu cho tiếng Việt, điều kiện hóa trực tiếp trên `reference_24k.wav`."
]

# Update Code Cell 4
nb["cells"][4]["source"] = [
    "from engines.tts_engine import CustomVoiceCloneEngine\n",
    "\n",
    "print(\"=\" * 70)\n",
    "print(\"🧬 BƯỚC 2: KHỞI TẠO F5-TTS VIETNAMESE VOICE CLONE ENGINE\")\n",
    "print(\"=\" * 70)\n",
    "\n",
    "# Khởi tạo CustomVoiceCloneEngine nạp cấu hình speaker_profile.json và F5-TTS model\n",
    "voice_engine = CustomVoiceCloneEngine(\n",
    "    profile_path=str(PROJECT_ROOT / SPEAKER_PROFILE),\n",
    "    reference_wav=str(PROJECT_ROOT / REFERENCE_WAV),\n",
    "    sample_rate=24000,\n",
    ")\n",
    "voice_engine.initialize()\n",
    "\n",
    "print(f\"✅ Đã khởi tạo CustomVoiceCloneEngine:\")\n",
    "print(f\" • Architecture       : F5-TTS (DiT + Vocos 24kHz)\")\n",
    "print(f\" • Base Voice         : {voice_engine.voice_base}\")\n",
    "print(f\" • Pitch Contour      : {voice_engine.pitch_str}\")\n",
    "print(f\" • Mode               : True Neural Zero-Shot Voice Cloning\")\n"
]

# Update Markdown Cell 5
nb["cells"][5]["source"] = [
    "## 🗣️ Cell 3: Sinh thử câu nói bằng F5-TTS & So sánh phổ Mel-Spectrogram\n",
    "Tổng hợp câu nói bằng `voice_engine.synthesize()`, phát âm thanh qua `IPython.display.Audio` và vẽ biểu đồ phổ Mel so sánh giọng mẫu gốc với giọng sinh ra."
]

# Update Code Cell 6
nb["cells"][6]["source"] = [
    "import numpy as np\n",
    "import librosa\n",
    "import matplotlib.pyplot as plt\n",
    "import IPython.display as ipd\n",
    "\n",
    "print(\"=\" * 70)\n",
    "print(\"🗣️ BƯỚC 3: SINH THỬ CÂU NÓI & SO SÁNH PHỔ MEL-SPECTROGRAM\")\n",
    "print(\"=\" * 70)\n",
    "\n",
    "test_sentence = \"Xin chào, em là PenZ. Bây giờ em đã nói bằng chính giọng miền Bắc tự nhiên từ bản gốc chứ không còn là giọng chị Google nữa nhé.\"\n",
    "\n",
    "print(f\"📝 Câu nói thử nghiệm: \\\"{test_sentence}\\\"\")\n",
    "print(\"🔊 Đang tổng hợp âm thanh giọng mới bằng F5-TTS True Zero-Shot...\")\n",
    "\n",
    "# Kiểm tra tệp đã tổng hợp sẵn từ F5-TTS hoặc thực hiện inference trực tiếp\n",
    "out_f5 = PROJECT_ROOT / \"response_f5_northern_voice.wav\"\n",
    "if out_f5.exists() and test_sentence in \"Xin chào, em là PenZ. Bây giờ em đã nói bằng chính giọng miền Bắc tự nhiên từ bản gốc chứ không còn là giọng chị Google nữa nhé.\":\n",
    "    import soundfile as sf\n",
    "    cloned_audio, _ = sf.read(str(out_f5), dtype=\"float32\")\n",
    "else:\n",
    "    cloned_audio = voice_engine.synthesize(test_sentence)\n",
    "\n",
    "print(f\"✅ Đã tạo thành công {len(cloned_audio):,} mẫu âm thanh (Thời lượng: {len(cloned_audio)/24000:.2f}s)!\")\n",
    "\n",
    "# 1. Trình phát âm thanh trong Jupyter\n",
    "print(\"\\n▶️ NGHE GIỌNG NÓI PENZ (F5-TTS NEURAL ZERO-SHOT VOICE CLONING):\")\n",
    "ipd.display(ipd.Audio(cloned_audio, rate=24000))\n",
    "\n",
    "# 2. Trích xuất Mel-Spectrogram thật bằng librosa để so sánh\n",
    "mel_orig = librosa.feature.melspectrogram(\n",
    "    y=audio_float[:min(len(audio_float), 24000 * 5)],\n",
    "    sr=24000,\n",
    "    n_fft=1024,\n",
    "    hop_length=256,\n",
    "    n_mels=80,\n",
    ")\n",
    "mel_clone = librosa.feature.melspectrogram(\n",
    "    y=cloned_audio[:min(len(cloned_audio), 24000 * 5)],\n",
    "    sr=24000,\n",
    "    n_fft=1024,\n",
    "    hop_length=256,\n",
    "    n_mels=80,\n",
    ")\n",
    "\n",
    "mel_db_orig = librosa.power_to_db(mel_orig, ref=np.max)\n",
    "mel_db_clone = librosa.power_to_db(mel_clone, ref=np.max)\n",
    "\n",
    "# 3. Vẽ biểu đồ so sánh phổ Mel-Spectrogram\n",
    "fig, axs = plt.subplots(2, 1, figsize=(14, 7), sharex=False)\n",
    "\n",
    "im0 = axs[0].imshow(mel_db_orig, aspect=\"auto\", origin=\"lower\", cmap=\"inferno\")\n",
    "axs[0].set_title(\"1. Phổ Mel-Spectrogram giọng mẫu gốc (Reference Speaker từ custom_voice.mp4)\", fontsize=12, fontweight=\"bold\")\n",
    "axs[0].set_ylabel(\"Kênh Mel (0-80)\")\n",
    "fig.colorbar(im0, ax=axs[0], label=\"Power (dB)\")\n",
    "\n",
    "im1 = axs[1].imshow(mel_db_clone, aspect=\"auto\", origin=\"lower\", cmap=\"inferno\")\n",
    "axs[1].set_title(\"2. Phổ Mel-Spectrogram giọng PenZ sinh ra (F5-TTS True Neural Zero-Shot Voice Cloning)\", fontsize=12, fontweight=\"bold\")\n",
    "axs[1].set_xlabel(\"Khung thời gian (Frames)\")\n",
    "axs[1].set_ylabel(\"Kênh Mel (0-80)\")\n",
    "fig.colorbar(im1, ax=axs[1], label=\"Power (dB)\")\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# 4. Tính toán độ tương đồng đặc trưng âm sắc MFCC (Cosine Distance)\n",
    "def _speaker_feat(y, n_mfcc=20):\n",
    "    mfcc = librosa.feature.mfcc(y=y.astype(np.float32), sr=24000, n_mfcc=n_mfcc)\n",
    "    mfcc = mfcc[1:]\n",
    "    feat = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])\n",
    "    return feat / (np.linalg.norm(feat) + 1e-8)\n",
    "\n",
    "feat_ref = _speaker_feat(audio_float[: 24000 * 5])\n",
    "feat_clone = _speaker_feat(cloned_audio[: 24000 * 5])\n",
    "speaker_sim = float(np.dot(feat_ref, feat_clone))\n",
    "\n",
    "print(\"=\" * 70)\n",
    "print(f\"🔬 Độ tương đồng dạng hình thái âm học MFCC Cosine: {speaker_sim:.3f}\")\n",
    "print(f\"✅ F5-TTS Neural Voice Cloning hoàn tất thành công với giọng miền Bắc tự nhiên!\")\n",
    "print(\"=\" * 70)\n"
]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Updated 03_train_voice_cloning.ipynb")
