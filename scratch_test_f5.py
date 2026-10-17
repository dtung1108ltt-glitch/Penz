import sys
import os
import time

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import soundfile as sf
import torch
import torchaudio

# Patch torchaudio.load and torchaudio.save to use soundfile
def _sf_load(uri, *args, **kwargs):
    data, sr = sf.read(uri, dtype="float32")
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        tensor = tensor.t()
    return tensor, sr

def _sf_save(uri, src, sample_rate, *args, **kwargs):
    if isinstance(src, torch.Tensor):
        src = src.detach().cpu().numpy()
    if src.ndim == 2:
        src = src.T
    sf.write(uri, src, sample_rate)

torchaudio.load = _sf_load
torchaudio.save = _sf_save

from f5_tts.api import F5TTS

print("Initializing F5TTS with model='F5TTS_Base'...")
t0 = time.time()
f5 = F5TTS(
    model="F5TTS_Base",
    ckpt_file="D:/PenZ/models/f5_tts_vi/model_latest.safetensors",
    vocab_file="D:/PenZ/models/f5_tts_vi/vocab.txt",
    ode_method="euler",
    device="cpu"
)
print(f"Loaded model in {time.time() - t0:.2f}s")

ref_file = "D:/PenZ/models/voice_profile/reference_24k.wav"
ref_text = "Giữa triệu người lướt qua vô tình ta đã tìm thấy nhau, đã nhiều lần em sai nhưng hôm nay em chắc mình đúng."
gen_text = "Xin chào, em là PenZ. Bây giờ em đã nói bằng chính giọng miền Bắc tự nhiên từ bản gốc chứ không còn là giọng chị Google nữa nhé."

print("Starting inference...")
t1 = time.time()
wav, sr, _ = f5.infer(
    ref_file=ref_file,
    ref_text=ref_text,
    gen_text=gen_text,
    speed=1.0,
    nfe_step=32,
    show_info=lambda *a, **k: None
)
t2 = time.time()
print(f"Inference complete in {t2 - t1:.2f}s, sample rate: {sr}, wav shape: {wav.shape if hasattr(wav, 'shape') else len(wav)}")

out_path = "D:/PenZ/response_f5_northern_voice.wav"
sf.write(out_path, wav, sr)
print(f"Saved audio to {out_path}, size: {os.path.getsize(out_path)} bytes")
