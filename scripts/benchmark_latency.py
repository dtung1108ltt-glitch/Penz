import asyncio
import io
import sys
import time
from pathlib import Path
import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.audio_config import AudioConfig
from config.model_config import ModelConfig
from core.vad_detector import VADDetector
from engines.stt_engine import get_stt_engine
from engines.llm_engine import get_llm_engine
from engines.tts_engine import get_tts_engine


async def benchmark_vad(iterations: int = 1000) -> float:
    """Measure DSP VAD frame execution time in milliseconds."""
    vad = VADDetector()
    dummy_frame = np.random.randn(512).astype(np.float32) * 0.05

    start = time.perf_counter()
    for _ in range(iterations):
        vad.process_frame(dummy_frame)
    duration = time.perf_counter() - start

    avg_ms = (duration / iterations) * 1000.0
    return avg_ms


def benchmark_stt(stt_engine, sample_rate: int = 16000) -> float:
    """Measure STT transcription latency for a 3-second audio buffer."""
    # 3 seconds of dummy audio
    dummy_audio = np.random.randn(sample_rate * 3).astype(np.float32) * 0.1

    start = time.perf_counter()
    _ = stt_engine.transcribe(dummy_audio, sample_rate)
    duration_ms = (time.perf_counter() - start) * 1000.0
    return duration_ms


async def benchmark_llm_ttfb(llm_engine, test_prompt: str = "Thời tiết hôm nay thế nào?") -> tuple[float, float]:
    """Measure LLM Time-To-First-Sentence (TTFB) and total response time."""
    start = time.perf_counter()
    ttfb = 0.0
    first_received = False

    async for chunk in llm_engine.generate_stream(test_prompt):
        if not first_received:
            ttfb = (time.perf_counter() - start) * 1000.0
            first_received = True

    total_ms = (time.perf_counter() - start) * 1000.0
    if not first_received:
        ttfb = total_ms
    return ttfb, total_ms


async def benchmark_tts_first_chunk(tts_engine, sample_text: str = "Thời tiết hôm nay rất đẹp và nắng ráo.") -> float:
    """Measure TTS first chunk synthesis latency."""
    start = time.perf_counter()
    first_received = False
    first_chunk_ms = 0.0

    async for chunk in tts_engine.synthesize_stream(sample_text):
        if not first_received:
            first_chunk_ms = (time.perf_counter() - start) * 1000.0
            first_received = True
            break

    if not first_received:
        first_chunk_ms = (time.perf_counter() - start) * 1000.0
    return first_chunk_ms


async def run_benchmark(use_mock: bool = False):
    print("=" * 65)
    print("🚀 BẮT ĐẦU ĐO ĐẠC LATENCY HỆ THỐNG VOICE AI (PIPELINE BENCHMARK)")
    print("=" * 65)

    audio_cfg = AudioConfig()
    model_cfg = ModelConfig()

    if use_mock:
        model_cfg.stt.backend = "mock"
        model_cfg.llm.provider = "mock"
        model_cfg.tts.backend = "mock"

    stt_engine = get_stt_engine(model_cfg.stt)
    try:
        stt_engine.initialize()
    except Exception as e:
        print(f"⚠️  STT engine ({model_cfg.stt.backend}) unavailable ({e}). Fallback to MockSTTEngine.")
        from engines.stt_engine import MockSTTEngine
        stt_engine = MockSTTEngine()
        stt_engine.initialize()

    llm_engine = get_llm_engine(model_cfg.llm)
    try:
        llm_engine.initialize()
    except Exception as e:
        print(f"⚠️  LLM engine ({model_cfg.llm.provider}) unavailable ({e}). Fallback to MockLLMEngine.")
        from engines.llm_engine import MockLLMEngine
        llm_engine = MockLLMEngine()
        llm_engine.initialize()

    tts_engine = get_tts_engine(model_cfg.tts)
    try:
        tts_engine.initialize()
    except Exception as e:
        print(f"⚠️  TTS engine ({model_cfg.tts.backend}) unavailable ({e}). Fallback to MockTTSEngine.")
        from engines.tts_engine import MockTTSEngine
        tts_engine = MockTTSEngine()
        tts_engine.initialize()

    # 1. VAD Latency
    vad_latency_ms = await benchmark_vad()
    print(f"1. VAD Frame Processing Latency (512 samples / 32ms frame): {vad_latency_ms:.3f} ms")

    # 2. STT Latency
    stt_latency_ms = benchmark_stt(stt_engine, audio_cfg.sample_rate)
    print(f"2. STT Latency (Transcribe 3.0s audio):                 {stt_latency_ms:.1f} ms")

    # 3. LLM TTFB
    llm_ttfb_ms, llm_total_ms = await benchmark_llm_ttfb(llm_engine)
    print(f"3. LLM Time-To-First-Byte (TTFB First Sentence):       {llm_ttfb_ms:.1f} ms")
    print(f"   LLM Total Generation Time:                          {llm_total_ms:.1f} ms")

    # 4. TTS First Chunk
    tts_first_chunk_ms = await benchmark_tts_first_chunk(tts_engine)
    print(f"4. TTS First Audio Chunk Latency:                      {tts_first_chunk_ms:.1f} ms")

    # 5. Total Voice-to-Voice Latency
    v2v_latency_ms = stt_latency_ms + llm_ttfb_ms + tts_first_chunk_ms

    print("=" * 65)
    print("📊 TỔNG KẾT BẢNG ĐO THỜI GIAN ĐÁP ỨNG (END-TO-END LATENCY SCORECARD)")
    print("=" * 65)
    print(f"{'Thành phần (Component)':<35} | {'Thời gian (Latency)':<15} | {'Mục tiêu tối ưu'}")
    print("-" * 65)
    print(f"{'VAD Frame Processing':<35} | {vad_latency_ms:>7.3f} ms     | < 1.0 ms")
    print(f"{'Speech-to-Text (STT)':<35} | {stt_latency_ms:>7.1f} ms     | < 400.0 ms")
    print(f"{'LLM Time-to-First-Byte (TTFB)':<35} | {llm_ttfb_ms:>7.1f} ms     | < 500.0 ms")
    print(f"{'TTS First Chunk Synthesis':<35} | {tts_first_chunk_ms:>7.1f} ms     | < 300.0 ms")
    print("-" * 65)
    print(f"{'TỔNG THỜI GIAN VOICE-TO-VOICE':<35} | {v2v_latency_ms:>7.1f} ms     | < 1200.0 ms")
    print("=" * 65)

    stt_engine.shutdown()
    llm_engine.shutdown()
    tts_engine.shutdown()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark pipeline latency and TTFB")
    parser.add_argument("--mock", action="store_true", help="Force benchmark using mock engines")
    args = parser.parse_args()
    asyncio.run(run_benchmark(use_mock=args.mock))


if __name__ == "__main__":
    main()
