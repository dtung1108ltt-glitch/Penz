"""Production entry point for Voice AI Assistant system."""
import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from config.audio_config import AudioConfig
from config.model_config import ModelConfig
from config.vad_config import VADConfig
from core.audio_player import AudioPlayer
from core.audio_recorder import AudioRecorder
from core.vad_detector import VADDetector
from engines.llm_engine import get_llm_engine
from engines.stt_engine import get_stt_engine
from engines.tts_engine import get_tts_engine
from pipeline.event_bus import EventBus, EventType
from pipeline.orchestrator import VoiceOrchestrator
from pipeline.session_manager import SessionManager

# Configure logging format
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("voice_ai_system")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-Time Voice AI Assistant (PenZ)")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock engine mode (zero external downloads or models required)",
    )
    parser.add_argument(
        "--stt",
        type=str,
        default=os.getenv("STT_BACKEND", "faster-whisper"),
        help="STT backend: faster-whisper, mock (default: faster-whisper)",
    )
    parser.add_argument(
        "--stt-model",
        type=str,
        default=os.getenv("STT_MODEL_SIZE", "base"),
        help="STT model size: tiny, base, small, medium, large-v3 (default: base)",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default=os.getenv("COMPUTE_TYPE", "int8"),
        help="Compute type for STT: int8, float16, float32 (default: int8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.getenv("DEVICE", "cpu"),
        help="Compute device for inference: cuda, cpu (default: cpu)",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default=os.getenv("LLM_PROVIDER", "local" if Path("models/penz_voice_llm_merged").exists() else "mock"),
        help="LLM provider: local, ollama, openai, mock (default: local)",
    )
    parser.add_argument(
        "--tts",
        type=str,
        default=os.getenv("TTS_BACKEND", "custom"),
        help="TTS backend: custom, edge-tts, piper, mock (default: custom)",
    )
    parser.add_argument(
        "--tts-voice",
        type=str,
        default=os.getenv("TTS_VOICE", None),
        help="TTS base voice (default: auto from speaker profile or vi-VN-HoaiMyNeural)",
    )
    parser.add_argument(
        "--voice-profile",
        type=str,
        default=os.getenv("VOICE_PROFILE_PATH", "models/voice_profile/reference_24k.wav"),
        help="Path to personalized voice reference (wav or profile json) (default: models/voice_profile/reference_24k.wav)",
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to an input audio file to run full pipeline: STT -> LLM -> TTS -> Output file",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Direct input text query to run LLM -> TTS -> Output file",
    )
    parser.add_argument(
        "--output-audio",
        type=str,
        default="response_custom.mp3",
        help="Path to save synthesized audio output (default: response_custom.mp3)",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Disable automatic speaker playback of output audio",
    )
    return parser.parse_args()


async def run_single_turn_pipeline(args: argparse.Namespace, model_cfg: ModelConfig, audio_cfg: AudioConfig) -> None:
    """Execute a single-turn end-to-end Voice AI cycle:
    Audio/File (STT) -> LLM Brain -> Voice Synthesis (TTS) -> Save file & Playback.
    """
    print("\n" + "=" * 65)
    print("🚀 PENZ VOICE AI: CHẠY KIỂM THỬ ĐƯỜNG ỐNG ĐƠN LƯỢT (SINGLE-TURN PIPELINE)")
    print("=" * 65)

    # 1. Initialize Engines
    print(f"⚙️  STT Engine:  {model_cfg.stt.backend} (size={model_cfg.stt.model_size}, device={model_cfg.stt.device}, compute={model_cfg.stt.compute_type})")
    print(f"⚙️  LLM Engine:  {model_cfg.llm.provider}")
    print(f"⚙️  TTS Engine:  {model_cfg.tts.backend}")
    print("-" * 65)

    stt_engine = get_stt_engine(model_cfg.stt)
    llm_engine = get_llm_engine(model_cfg.llm)
    tts_engine = get_tts_engine(model_cfg.tts)

    # 2. STT: Nhận diện giọng nói
    user_query = ""
    if args.file:
        audio_path = Path(args.file)
        if not audio_path.exists():
            print(f"❌ Không tìm thấy tệp âm thanh đầu vào: {audio_path}")
            return
        print(f"🎙️  [BƯỚC 1: STT] Đang nhận diện giọng nói từ tệp: '{audio_path}'...")
        stt_engine.initialize()
        user_query = stt_engine.transcribe(str(audio_path))
        print(f"   👤 Người dùng nói (STT): \"{user_query}\"")
    elif args.text:
        user_query = args.text.strip()
        print(f"📝 [BƯỚC 1: TEXT] Nhận câu hỏi văn bản trực tiếp: \"{user_query}\"")
    else:
        print("❌ Vui lòng chỉ định --file <path> hoặc --text <câu hỏi>.")
        return

    if not user_query:
        print("⚠️ Không nhận diện được nội dung hoặc câu hỏi rỗng.")
        return

    # 3. LLM: Bộ não AI suy luận
    print(f"\n🧠 [BƯỚC 2: LLM] Bộ não AI ({model_cfg.llm.provider}) đang suy luận...")
    llm_engine.initialize()
    ai_response = llm_engine.generate(user_query)
    print(f"   🤖 Trợ lý PenZ phản hồi: \"{ai_response}\"")

    # 4. TTS: Tổng hợp giọng nói
    output_audio_path = Path(args.output_audio)
    tts_engine.initialize()
    if hasattr(tts_engine, "profile") and isinstance(tts_engine.profile, dict):
        spk = tts_engine.profile.get("speaker_name", "custom_voice")
        f0 = tts_engine.profile.get("f0_median_hz", "N/A")
        print(f"\n🔊 [BƯỚC 3: TTS] Đang tổng hợp giọng nói cá nhân hóa (Profile: {spk}, F0: {f0}Hz, Shift: {getattr(tts_engine, 'pitch_str', '+0Hz')})...")
        print(f"   🎯 Tệp tham chiếu: {getattr(tts_engine, 'reference_wav', 'N/A')}")
    else:
        print(f"\n🔊 [BƯỚC 3: TTS] Đang tổng hợp giọng nói qua {model_cfg.tts.backend} ({getattr(model_cfg.tts, 'voice', 'Default')})...")

    if hasattr(tts_engine, "synthesize_to_file"):
        saved_file = tts_engine.synthesize_to_file(ai_response, str(output_audio_path))
    else:
        import soundfile as sf
        audio_data = tts_engine.synthesize(ai_response)
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_audio_path), audio_data, 24000)
        saved_file = str(output_audio_path)
    print(f"   💾 Tệp âm thanh đã được lưu thành công tại: {saved_file}")

    # 5. Playback
    if not args.no_play:
        print("\n🎧 [BƯỚC 4: PLAYBACK] Đang phát âm thanh phản hồi qua loa...")
        try:
            player = AudioPlayer(audio_cfg)
            player.start()
            if hasattr(tts_engine, "synthesize"):
                audio_wave = tts_engine.synthesize(ai_response)
                player.play(audio_wave)
            player.stop()
            print("   ✅ Đã phát âm thanh xong.")
        except Exception as e:
            logger.warning("Không thể phát âm thanh trực tiếp: %s. Bạn có thể mở tệp %s.", e, saved_file)

    print("=" * 65)
    print("🎉 HOÀN THÀNH TOÀN BỘ ĐƯỜNG ỐNG: STT -> LLM -> TTS -> AUDIO FILE!")
    print("=" * 65)


async def run_system(args: argparse.Namespace) -> None:
    """Build dependency graph and launch orchestrator loop."""
    print("=" * 65)
    print("🎙️  PRODUCTION-GRADE VOICE AI ASSISTANT SYSTEM (PENZ)")
    print("=" * 65)

    # 1. Initialize Configurations
    audio_cfg = AudioConfig()
    vad_cfg = VADConfig()
    model_cfg = ModelConfig()

    if args.mock:
        model_cfg.stt.backend = "mock"
        model_cfg.llm.provider = "mock"
        model_cfg.tts.backend = "mock"
    else:
        model_cfg.stt.backend = args.stt
        model_cfg.stt.model_size = args.stt_model
        model_cfg.stt.compute_type = args.compute_type
        model_cfg.stt.device = args.device
        model_cfg.llm.provider = args.llm
        model_cfg.tts.backend = args.tts
        if args.tts_voice:
            model_cfg.tts.voice = args.tts_voice

        # Cấu hình hồ sơ giọng nói cá nhân hóa
        if hasattr(args, "voice_profile") and args.voice_profile:
            vp = Path(args.voice_profile)
            if vp.suffix.lower() in (".wav", ".mp3", ".flac"):
                model_cfg.tts.reference_wav_path = str(vp)
                json_cand = vp.parent / "speaker_profile.json"
                model_cfg.tts.voice_profile_path = str(json_cand if json_cand.exists() else vp)
            elif vp.suffix.lower() == ".json":
                model_cfg.tts.voice_profile_path = str(vp)
                model_cfg.tts.reference_wav_path = str(vp.parent / "reference_24k.wav")
            else:
                model_cfg.tts.voice_profile_path = str(vp / "speaker_profile.json")
                model_cfg.tts.reference_wav_path = str(vp / "reference_24k.wav")

    # Nếu người dùng truyền --file hoặc --text, chạy đường ống kiểm thử đơn lượt
    if args.file or args.text:
        await run_single_turn_pipeline(args, model_cfg, audio_cfg)
        return

    print(f"• Audio Config: {audio_cfg.sample_rate}Hz | Chunk: {audio_cfg.chunk_size} samples")
    print(f"• STT Backend:  {model_cfg.stt.backend} ({model_cfg.stt.model_size}) on {model_cfg.stt.device} ({model_cfg.stt.compute_type})")
    print(f"• LLM Engine:   {model_cfg.llm.provider}")
    print(f"• TTS Backend:  {model_cfg.tts.backend} (profile: {model_cfg.tts.reference_wav_path})")
    print("=" * 65)

    # 2. Layer 1: Core DSP (Hardware I/O & Signal Processing)
    recorder = AudioRecorder(audio_cfg, pre_buffer_duration_ms=vad_cfg.pre_buffer_duration_ms)
    player = AudioPlayer(audio_cfg)
    vad = VADDetector(vad_cfg, sample_rate=audio_cfg.sample_rate, chunk_size=audio_cfg.chunk_size)

    # 3. Layer 2: Inference Engines (Dependency Inversion)
    stt_engine = get_stt_engine(model_cfg.stt)
    llm_engine = get_llm_engine(model_cfg.llm)
    tts_engine = get_tts_engine(model_cfg.tts)

    # 4. Layer 3: System Pipeline Orchestration
    event_bus = EventBus()
    session_manager = SessionManager(max_turns=10)

    orchestrator = VoiceOrchestrator(
        recorder=recorder,
        player=player,
        vad=vad,
        stt_engine=stt_engine,
        llm_engine=llm_engine,
        tts_engine=tts_engine,
        session_manager=session_manager,
        event_bus=event_bus,
        audio_config=audio_cfg,
    )

    # Setup graceful shutdown handler
    stop_event = asyncio.Event()

    def handle_sigint():
        logger.info("Signal received. Triggering graceful shutdown...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_sigint)
        except NotImplementedError:
            pass

    orchestrator_task = asyncio.create_task(orchestrator.start())

    try:
        await orchestrator_task
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received.")
    finally:
        await orchestrator.stop()


def main():
    args = parse_arguments()
    try:
        asyncio.run(run_system(args))
    except KeyboardInterrupt:
        print("\n👋 Ứng dụng đã dừng thành công.")


if __name__ == "__main__":
    main()
