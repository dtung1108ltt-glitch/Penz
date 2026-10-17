import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
from pathlib import Path

# Let's test running the Python code of notebook 04 directly in a script to see what happens
print("Testing Cell 2 execution directly in Python...")
PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoModelForCausalLM, AutoTokenizer
from engines.stt_engine import FasterWhisperSTTEngine
from engines.tts_engine import CustomVoiceCloneEngine
from config.model_config import STTConfig, TTSConfig
import torch

MERGED_LLM_PATH = "models/penz_voice_llm_merged"
VOICE_PROFILE_PATH = "models/voice_profile/speaker_profile.json"
REFERENCE_WAV_PATH = "models/voice_profile/reference_24k.wav"

print("1. STT...")
stt_config = STTConfig(model_size="base", device="cpu", compute_type="int8", language="vi")
stt_engine = FasterWhisperSTTEngine(stt_config)
stt_engine.initialize()
print("STT ready")

print("2. LLM...")
llm_path = (PROJECT_ROOT / MERGED_LLM_PATH).resolve()
llm_tokenizer = AutoTokenizer.from_pretrained(str(llm_path), trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    str(llm_path),
    dtype=torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
)
print("LLM ready")

print("3. TTS...")
tts_engine = CustomVoiceCloneEngine(
    profile_path=str(PROJECT_ROOT / VOICE_PROFILE_PATH),
    reference_wav=str(PROJECT_ROOT / REFERENCE_WAV_PATH),
    sample_rate=24000,
)
tts_engine.initialize()
print("TTS ready")

print("4. Testing LLM generation...")
from scripts.build_llm_algorithms_corpus import SYSTEM_PROMPT
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Hãy giải thích cơ chế Rotary Position Embedding (RoPE) trong LLM."},
]
prompt = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm_model.device)

with torch.no_grad():
    outputs = llm_model.generate(
        **inputs,
        max_new_tokens=40,
        temperature=0.3,
        do_sample=True,
        pad_token_id=llm_tokenizer.eos_token_id,
    )
answer = llm_tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()
print("LLM Answer:", answer)
