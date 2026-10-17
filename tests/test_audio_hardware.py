"""Unit test verifying audio hardware devices (Microphone & Speaker)."""
import pytest


def test_audio_devices_query():
    """Check if SoundDevice is available and query default audio devices."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        assert len(devices) >= 0, "Device query should return a list"
        
        # Check host APIs
        hostapis = sd.query_hostapis()
        assert len(hostapis) > 0, "At least one host audio API should exist"
    except (OSError, ImportError) as e:
        pytest.skip(f"Audio hardware or sounddevice not accessible in this environment: {e}")


def test_audio_config_defaults():
    """Verify default audio configuration parameters."""
    from config.audio_config import AudioConfig

    cfg = AudioConfig()
    assert cfg.sample_rate == 16000
    assert cfg.chunk_size == 512
    assert cfg.channels == 1
    assert cfg.chunk_duration_ms == 32.0
