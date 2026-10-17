"""End-to-End integration test simulating the entire Voice AI loop without physical hardware."""
import asyncio
import numpy as np

from config.audio_config import AudioConfig
from config.vad_config import VADConfig
from core.audio_player import AudioPlayer
from core.audio_recorder import AudioRecorder
from core.vad_detector import VADDetector
from engines.llm_engine import MockLLMEngine
from engines.stt_engine import MockSTTEngine
from engines.tts_engine import MockTTSEngine
from pipeline.event_bus import EventBus, EventType
from pipeline.orchestrator import VoiceOrchestrator
from pipeline.session_manager import SessionManager


def test_end_to_end_mock_turn():
    """Verify complete conversational turn with mock engines."""
    async def _run():
        event_bus = EventBus()
        received_events = []

        def record_event(evt):
            return lambda **kw: received_events.append(evt)

        for evt in EventType:
            event_bus.subscribe(evt, record_event(evt))

        session = SessionManager()
        stt = MockSTTEngine(mock_response="Thời tiết hôm nay thế nào?")
        llm = MockLLMEngine(response_text="Hôm nay trời nắng đẹp.")
        tts = MockTTSEngine()
        vad = VADDetector(VADConfig(silence_threshold_db=-40.0, min_speech_duration_ms=32, min_silence_duration_ms=32))

        audio_cfg = AudioConfig()
        recorder = AudioRecorder(audio_cfg)
        player = AudioPlayer(audio_cfg)

        stt.initialize()
        llm.initialize()
        tts.initialize()

        speech_audio = np.ones(16000, dtype=np.float32) * 0.1

        orchestrator = VoiceOrchestrator(
            recorder=recorder,
            player=player,
            vad=vad,
            stt_engine=stt,
            llm_engine=llm,
            tts_engine=tts,
            session_manager=session,
            event_bus=event_bus,
        )

        orchestrator._is_running = True
        await orchestrator._process_ai_response(speech_audio)

        history = session.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Thời tiết hôm nay thế nào?"
        assert history[1]["role"] == "assistant"

        assert EventType.STT_TRANSCRIBED in received_events
        assert EventType.AI_THINKING_START in received_events
        assert EventType.AI_RESPONSE_COMPLETE in received_events

    asyncio.run(_run())


def test_barge_in_interruption():
    """Test that barge-in immediately interrupts active audio playback."""
    async def _run():
        audio_cfg = AudioConfig()
        player = AudioPlayer(audio_cfg)
        event_bus = EventBus()
        interrupt_fired = False

        async def on_interrupt(**kw):
            nonlocal interrupt_fired
            interrupt_fired = True

        event_bus.subscribe(EventType.INTERRUPT, on_interrupt)

        chunk = np.ones(1024, dtype=np.float32) * 0.1
        player.play_chunk(chunk)
        assert player.is_playing

        orchestrator = VoiceOrchestrator(
            recorder=AudioRecorder(audio_cfg),
            player=player,
            vad=VADDetector(),
            stt_engine=MockSTTEngine(),
            llm_engine=MockLLMEngine(),
            tts_engine=MockTTSEngine(),
            event_bus=event_bus,
        )

        await orchestrator._handle_barge_in()

        assert not player.is_playing
        assert interrupt_fired

    asyncio.run(_run())
