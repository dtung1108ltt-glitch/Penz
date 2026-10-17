"""Unit tests for STT engine interfaces and word error rate metrics."""
import numpy as np
import pytest

from config.model_config import STTConfig
from engines.base_engine import BaseSTTEngine
from engines.stt_engine import MockSTTEngine, get_stt_engine
from training.stt_finetune.evaluate_wer import compute_wer, compute_cer


def test_stt_engine_interface():
    """Verify STT engine contract compliance."""
    engine = MockSTTEngine(mock_response="Xin chào Việt Nam")
    assert isinstance(engine, BaseSTTEngine)
    assert not engine.is_ready

    engine.initialize()
    assert engine.is_ready

    audio_sample = np.zeros(16000, dtype=np.float32)
    text = engine.transcribe(audio_sample)
    assert text == "Xin chào Việt Nam"

    engine.shutdown()
    assert not engine.is_ready


def test_stt_factory():
    """Verify factory returns appropriate engine instance."""
    cfg = STTConfig(backend="mock")
    engine = get_stt_engine(cfg)
    assert isinstance(engine, MockSTTEngine)


def test_error_rate_metrics():
    """Test WER and CER calculations."""
    ref = "Xin chào các bạn"
    hyp_identical = "Xin chào các bạn"
    assert compute_wer(ref, hyp_identical) == 0.0
    assert compute_cer(ref, hyp_identical) == 0.0

    hyp_sub = "Xin chào quý bạn"
    assert compute_wer(ref, hyp_sub) == 0.25  # 1 substitution out of 4 words
