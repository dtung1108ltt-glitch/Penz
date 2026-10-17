"""Unit test for VAD DSP metrics and RingBuffer circular queue."""
import numpy as np
import pytest

from config.vad_config import VADConfig
from core.ring_buffer import RingBuffer
from core.vad_detector import VADDetector


def test_ring_buffer_fifo():
    """Verify that RingBuffer stores and retrieves data in exact chronological order."""
    capacity = 100
    rb = RingBuffer(capacity_samples=capacity, dtype=np.float32)

    # Write 50 samples
    first_chunk = np.arange(50, dtype=np.float32)
    rb.write(first_chunk)
    assert len(rb) == 50
    np.testing.assert_array_equal(rb.read_all(), first_chunk)

    # Write 80 samples -> total 130 -> should retain the latest 100
    second_chunk = np.arange(50, 130, dtype=np.float32)
    rb.write(second_chunk)
    assert len(rb) == 100

    expected = np.arange(30, 130, dtype=np.float32)
    np.testing.assert_array_equal(rb.read_all(), expected)

    # Clear
    rb.clear()
    assert len(rb) == 0
    assert len(rb.read_all()) == 0


def test_vad_dsp_metrics():
    """Verify RMS dB and ZCR on synthetic waveforms."""
    # 1. Total silence
    silence = np.zeros(512, dtype=np.float32)
    assert VADDetector.calculate_rms(silence) == 0.0
    assert VADDetector.calculate_rms_db(silence) <= -90.0
    assert VADDetector.calculate_zcr(silence) == 0.0

    # 2. Known 440 Hz Sine Wave at peak 0.5
    sr = 16000
    t = np.linspace(0, 512 / sr, 512, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    rms = VADDetector.calculate_rms(sine)
    # Theoretical RMS of sine with amplitude A is A / sqrt(2) = 0.5 / 1.414 ~ 0.353
    assert 0.34 < rms < 0.36

    db = VADDetector.calculate_rms_db(sine)
    assert -10.0 < db < -8.0

    zcr = VADDetector.calculate_zcr(sine)
    assert zcr > 0.02


def test_vad_state_machine_transitions():
    """Test full speech cycle: Silence -> Speech Start -> Speaking -> Speech End."""
    cfg = VADConfig(
        silence_threshold_db=-35.0,
        zcr_threshold=0.02,
        min_speech_duration_ms=64,   # 2 frames @ 32ms
        min_silence_duration_ms=64,  # 2 frames @ 32ms
    )
    vad = VADDetector(config=cfg, sample_rate=16000, chunk_size=512)

    silence_frame = np.zeros(512, dtype=np.float32)
    # Loud sine wave
    t = np.linspace(0, 512 / 16000, 512, endpoint=False)
    speech_frame = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    # Initial state is silence
    res = vad.process_frame(silence_frame)
    assert not res.is_speech
    assert not res.speech_start

    # Frame 1 of speech: potential speech
    res1 = vad.process_frame(speech_frame)
    # Frame 2 of speech: triggers speech_start (since min_speech_duration_ms = 64ms)
    res2 = vad.process_frame(speech_frame)
    assert res2.speech_start or res1.speech_start or res2.is_speech

    # Frame 3 of speech: sustained speech
    res3 = vad.process_frame(speech_frame)
    assert res3.is_speech
    assert not res3.speech_start

    # Silence frame 1
    _ = vad.process_frame(silence_frame)
    # Silence frame 2 -> triggers speech_end
    res_end = vad.process_frame(silence_frame)
    assert res_end.speech_end or not res_end.is_speech
