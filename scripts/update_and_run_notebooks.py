"""Update notebooks with new reference slice and execute all notebooks to refresh cell outputs."""
import sys
import io
from pathlib import Path
import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 1. Update Cell 2 in notebooks/01_data_and_voice_prep.ipynb
nb1_path = PROJECT_ROOT / "notebooks" / "01_data_and_voice_prep.ipynb"
nb1 = nbformat.read(str(nb1_path), as_version=4)

cell2_src = """import sys
import json
from pathlib import Path
import numpy as np

# Thiết lập Project Root ở dòng đầu tiên
PROJECT_ROOT = Path("..").resolve() if Path("..").joinpath("scripts").exists() else Path(".").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Cấu hình UTF-8 cho môi trường Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from scripts.enroll_custom_voice import enroll_voice

# ==============================================================================
# ⚙️ CẤU HÌNH ĐƯỜNG DẪN TỆP NGUỒN & ĐÍCH
# ==============================================================================
INPUT_VIDEO_PATH = "data/raw/custom_voice.mp4"
OUTPUT_WAV_PATH = "models/voice_profile/reference_24k.wav"
OUTPUT_PROFILE_PATH = "models/voice_profile/speaker_profile.json"

# Cắt phân đoạn chuẩn 28.0s -> 38.5s (câu hát rõ chữ, nhịp đều, tránh ngân dài)
START_SEC = 28.0
END_SEC = 38.5
REF_TEXT = "Giữa triệu người lướt qua vô tình ta đã tìm thấy nhau, đã nhiều lần em sai nhưng hôm nay em chắc mình đúng."

input_file = (PROJECT_ROOT / INPUT_VIDEO_PATH).resolve()
output_wav = (PROJECT_ROOT / OUTPUT_WAV_PATH).resolve()
output_profile = (PROJECT_ROOT / OUTPUT_PROFILE_PATH).resolve()

print("=" * 70)
print("🎙️ BƯỚC 1: TRÍCH XUẤT GIỌNG NÓI TỪ VIDEO & ĐĂNG KÝ HỒ SƠ ÂM HỌC")
print("=" * 70)
print(f"• Video nguồn       : {input_file}")
print(f"• Tệp WAV đích      : {output_wav}")
print(f"• Hồ sơ người nói   : {output_profile}")
print(f"• Phân đoạn cắt     : {START_SEC}s -> {END_SEC}s ({END_SEC - START_SEC:.1f}s)")
print(f"• Transcript chuẩn  : {REF_TEXT}")

if not input_file.exists():
    raise FileNotFoundError(f"Không tìm thấy file video mẫu tại: {input_file}")

# Thực hiện trích xuất, cắt chuẩn và chuẩn hóa về 24kHz Mono PCM
audio_24k, profile = enroll_voice(
    input_file=input_file,
    output_wav_path=output_wav,
    profile_json_path=output_profile,
    target_sample_rate=24000,
    start_sec=START_SEC,
    end_sec=END_SEC,
    ref_text=REF_TEXT,
    target_speaking_f0=230.0
)

print("\\n" + "=" * 70)
print("✅ KẾT QUẢ ĐĂNG KÝ HỒ SƠ GIỌNG NÓI THÀNH CÔNG:")
print("=" * 70)
print(f"👤 Tên người nói       : {profile['speaker_name']}")
print(f"⏱️ Thời lượng âm thanh  : {profile['duration_seconds']:.2f} giây ({len(audio_24k):,} mẫu)")
print(f"🎵 F0 Trung vị (Median) : {profile['f0_median_hz']:.1f} Hz")
print(f"🗣️ F0 Quãng nói tự nhiên: {profile.get('target_speaking_f0_hz', 230.0):.1f} Hz")
print(f"📈 F0 Trung bình (Mean) : {profile['f0_mean_hz']:.1f} Hz (Độ lệch chuẩn: {profile['f0_std_hz']:.1f} Hz)")
print(f"🗣️ Tỷ lệ khung hữu thanh: {profile['voiced_frame_ratio']*100:.1f}%")
print(f"🔊 Năng lượng RMS       : {profile['rms_energy']:.4f}")
print(f"🏷️ Phân loại dải giọng  : {profile['estimated_pitch_band']}")
print(f"🎚️ Độ dịch pitch gợi ý  : {float(profile['recommended_pitch_shift_semitones']):+.1f} semitones")
print(f"📜 Câu mẫu tham chiếu   : \\\"{profile.get('ref_text', '')}\\\"")
print("=" * 70)
"""

nb1.cells[2].source = cell2_src
nbformat.write(nb1, str(nb1_path))
print("Updated 01_data_and_voice_prep.ipynb source!")

# 2. Update Cell 6 test sentence in notebooks/03_train_voice_cloning.ipynb
nb3_path = PROJECT_ROOT / "notebooks" / "03_train_voice_cloning.ipynb"
nb3 = nbformat.read(str(nb3_path), as_version=4)

cell6_src = """import numpy as np
import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd

print("=" * 70)
print("🗣️ BƯỚC 3: SINH THỬ CÂU NÓI & SO SÁNH PHỔ MEL-SPECTROGRAM")
print("=" * 70)

test_sentence = "Xin chào, em là PenZ. Hôm nay anh có điều gì muốn trò chuyện cùng em không?"

print(f"📝 Câu nói thử nghiệm: \\\"{test_sentence}\\\"")
print("🔊 Đang tổng hợp âm thanh giọng mới...")

cloned_audio = voice_engine.synthesize(test_sentence)
print(f"✅ Đã tạo thành công {len(cloned_audio):,} mẫu âm thanh (Thời lượng: {len(cloned_audio)/24000:.2f}s)!")

# 1. Trình phát âm thanh trong Jupyter
print("\\n▶️ NGHE GIỌNG NÓI PENZ (PITCH/RATE ĐÃ ĐIỀU CHỈNH):")
ipd.display(ipd.Audio(cloned_audio, rate=24000))

# 2. Trích xuất Mel-Spectrogram thật bằng librosa để so sánh
mel_orig = librosa.feature.melspectrogram(
    y=audio_float[:min(len(audio_float), 24000 * 5)],
    sr=24000,
    n_fft=1024,
    hop_length=256,
    n_mels=80,
)
mel_clone = librosa.feature.melspectrogram(
    y=cloned_audio[:min(len(cloned_audio), 24000 * 5)],
    sr=24000,
    n_fft=1024,
    hop_length=256,
    n_mels=80,
)

mel_db_orig = librosa.power_to_db(mel_orig, ref=np.max)
mel_db_clone = librosa.power_to_db(mel_clone, ref=np.max)

# 3. Vẽ biểu đồ so sánh phổ Mel-Spectrogram
fig, axs = plt.subplots(2, 1, figsize=(14, 7), sharex=False)

im0 = axs[0].imshow(mel_db_orig, aspect="auto", origin="lower", cmap="inferno")
axs[0].set_title("1. Phổ Mel-Spectrogram giọng mẫu gốc (Reference Speaker từ custom_voice.mp4)", fontsize=12, fontweight="bold")
axs[0].set_ylabel("Kênh Mel (0-80)")
fig.colorbar(im0, ax=axs[0], label="Power (dB)")

im1 = axs[1].imshow(mel_db_clone, aspect="auto", origin="lower", cmap="inferno")
axs[1].set_title("2. Phổ Mel-Spectrogram giọng PenZ sinh ra (Pitch/Rate Calibrated & Formant Shifting)", fontsize=12, fontweight="bold")
axs[1].set_xlabel("Khung thời gian (Frames)")
axs[1].set_ylabel("Kênh Mel (0-80)")
fig.colorbar(im1, ax=axs[1], label="Power (dB)")

plt.tight_layout()
plt.show()

# 4. Tính toán độ tương đồng đặc trưng âm sắc MFCC (Cosine Distance)
def _speaker_feat(y, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=y.astype(np.float32), sr=24000, n_mfcc=n_mfcc)
    mfcc = mfcc[1:]
    feat = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
    return feat / (np.linalg.norm(feat) + 1e-8)

feat_ref = _speaker_feat(audio_float[: 24000 * 5])
feat_clone = _speaker_feat(cloned_audio[: 24000 * 5])
speaker_sim = float(np.dot(feat_ref, feat_clone))

print("=" * 70)
print(f"🔬 Độ tương đồng dạng hình thái âm học MFCC Cosine: {speaker_sim:.3f}")
print(f"✅ Tinh chỉnh giọng nói hoàn tất thành công!")
print("=" * 70)
"""

nb3.cells[6].source = cell6_src
nbformat.write(nb3, str(nb3_path))
nbformat.write(nb3, str(PROJECT_ROOT / "notebooks" / "03_voice_pitch_rate_tuning.ipynb"))
print("Updated 03_train_voice_cloning.ipynb source!")

# 3. Execute all notebooks sequentially to refresh cell outputs
notebooks_to_run = [
    "01_data_and_voice_prep.ipynb",
    "03_train_voice_cloning.ipynb",
    "04_penz_interactive_studio.ipynb",
]

for nb_name in notebooks_to_run:
    p = PROJECT_ROOT / "notebooks" / nb_name
    print(f"Executing {nb_name}...")
    nb_obj = nbformat.read(str(p), as_version=4)
    client = NotebookClient(nb_obj, timeout=600, kernel_name="python3")
    client.execute()
    nbformat.write(nb_obj, str(p))
    if nb_name == "03_train_voice_cloning.ipynb":
        nbformat.write(nb_obj, str(PROJECT_ROOT / "notebooks" / "03_voice_pitch_rate_tuning.ipynb"))
    print(f"-> {nb_name} executed and saved successfully!")

print("All notebooks updated and executed successfully!")
