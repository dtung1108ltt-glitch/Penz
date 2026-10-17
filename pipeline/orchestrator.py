"""Real-time Voice Pipeline Orchestrator coordinating DSP Core and Inference Engines."""
import asyncio
import logging
from typing import Optional
import numpy as np

from config.audio_config import AudioConfig
from core.audio_player import AudioPlayer
from core.audio_recorder import AudioRecorder
from core.vad_detector import VADDetector
from engines.base_engine import BaseLLMEngine, BaseSTTEngine, BaseTTSEngine
from pipeline.event_bus import EventBus, EventType
from pipeline.session_manager import SessionManager

logger = logging.getLogger(__name__)


class VoiceOrchestrator:
    """Production coordinator running the full real-time Voice-to-Voice cycle.
    
    Coordinates: AudioRecorder -> VAD -> STT -> Session -> LLM -> TTS -> AudioPlayer
    Enforces instant interruption (barge-in) and clean asynchronous lifecycle.
    """

    def __init__(
        self,
        recorder: AudioRecorder,
        player: AudioPlayer,
        vad: VADDetector,
        stt_engine: BaseSTTEngine,
        llm_engine: BaseLLMEngine,
        tts_engine: BaseTTSEngine,
        session_manager: Optional[SessionManager] = None,
        event_bus: Optional[EventBus] = None,
        audio_config: Optional[AudioConfig] = None,
    ):
        self.recorder = recorder
        self.player = player
        self.vad = vad
        self.stt_engine = stt_engine
        self.llm_engine = llm_engine
        self.tts_engine = tts_engine
        self.session_manager = session_manager if session_manager is not None else SessionManager()
        self.event_bus = event_bus if event_bus is not None else EventBus()
        self.audio_config = audio_config if audio_config is not None else AudioConfig()

        self._is_running = False
        self._current_speech_chunks: list[np.ndarray] = []
        self._ai_turn_task: Optional[asyncio.Task] = None
        self._is_ai_generating = False

    async def start(self) -> None:
        """Start the real-time voice processing loop."""
        if self._is_running:
            return

        self._is_running = True
        logger.info("Initializing inference engines...")
        self.stt_engine.initialize()
        self.llm_engine.initialize()
        self.tts_engine.initialize()

        logger.info("Starting audio hardware streams...")
        self.recorder.start()
        self.player.start()

        logger.info("Voice Assistant is listening. Speak into your microphone.")

        try:
            await self._orchestration_loop()
        except asyncio.CancelledError:
            logger.info("Orchestration loop cancelled.")
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Gracefully stop all pipeline components and release resources."""
        if not self._is_running:
            return

        self._is_running = False
        logger.info("Stopping Voice Orchestrator...")

        if self._ai_turn_task and not self._ai_turn_task.done():
            self._ai_turn_task.cancel()

        self.recorder.stop()
        self.player.stop()

        self.stt_engine.shutdown()
        self.llm_engine.shutdown()
        self.tts_engine.shutdown()
        logger.info("Voice Orchestrator stopped cleanly.")

    async def _handle_barge_in(self) -> None:
        """Immediately abort AI speech and generation when user starts talking."""
        if self.player.is_playing or self._is_ai_generating:
            logger.info("⚡ [BARGE-IN] User interruption detected! Halting AI playback immediately.")
            # 1. Abort player output buffer
            self.player.interrupt()
            # 2. Cancel running LLM/TTS background task
            if self._ai_turn_task and not self._ai_turn_task.done():
                self._ai_turn_task.cancel()
            self._is_ai_generating = False
            # 3. Publish interrupt event
            await self.event_bus.publish(EventType.INTERRUPT)

    async def _orchestration_loop(self) -> None:
        """Main event loop capturing microphone chunks, evaluating VAD, and routing turns."""
        loop = asyncio.get_running_loop()

        while self._is_running:
            # Non-blocking fetch of audio frame from recorder
            frame = await loop.run_in_executor(None, self.recorder.read_frame, 0.05)
            if frame is None:
                await asyncio.sleep(0.005)
                continue

            vad_result = self.vad.process_frame(frame)

            if vad_result.speech_start:
                logger.debug("Speech start detected (RMS: %.1f dB, ZCR: %.2f)", vad_result.rms_db, vad_result.zcr)
                await self._handle_barge_in()
                await self.event_bus.publish(EventType.USER_SPEAKING_START)

                # Initialize speech buffer with pre-buffer to preserve onset consonants
                pre_buffer = self.recorder.get_pre_buffer()
                self._current_speech_chunks = [pre_buffer, frame]

            elif vad_result.is_speech:
                self._current_speech_chunks.append(frame)

            elif vad_result.speech_end:
                logger.debug("Speech end detected. Processing complete utterance...")
                await self.event_bus.publish(EventType.USER_SPEAKING_STOP)

                if self._current_speech_chunks:
                    complete_audio = np.concatenate(self._current_speech_chunks)
                    self._current_speech_chunks = []

                    # Launch AI processing task
                    self._ai_turn_task = asyncio.create_task(self._process_ai_response(complete_audio))

    async def _process_ai_response(self, audio_data: np.ndarray) -> None:
        """Transcribe user audio, query LLM stream, and synthesize TTS audio to player."""
        loop = asyncio.get_running_loop()
        self._is_ai_generating = True

        try:
            # 1. Speech-to-Text
            logger.info("Transcribing speech (%d samples)...", len(audio_data))
            text = await loop.run_in_executor(
                None, self.stt_engine.transcribe, audio_data, self.audio_config.sample_rate
            )
            text = text.strip()

            if not text:
                logger.debug("Empty transcription (silence/noise). Ignored.")
                return

            logger.info("👤 User: '%s'", text)
            await self.event_bus.publish(EventType.STT_TRANSCRIBED, text=text)
            self.session_manager.add_user_message(text)

            # 2. LLM Streaming & TTS Synthesis
            logger.info("🤖 AI thinking & streaming response...")
            await self.event_bus.publish(EventType.AI_THINKING_START)

            full_reply_parts = []
            is_first_sentence = True

            async for sentence in self.llm_engine.generate_stream(
                text, history=self.session_manager.get_history()
            ):
                if not self._is_running:
                    break

                clean_sentence = sentence.strip()
                if not clean_sentence:
                    continue

                full_reply_parts.append(clean_sentence)
                await self.event_bus.publish(EventType.AI_RESPONSE_CHUNK, chunk=clean_sentence)
                logger.info("💬 AI chunk: '%s'", clean_sentence)

                if is_first_sentence:
                    await self.event_bus.publish(EventType.AI_SPEAKING_START)
                    is_first_sentence = False

                # 3. Stream TTS for this sentence directly to audio player
                async for audio_chunk in self.tts_engine.synthesize_stream(clean_sentence):
                    if not self._is_running:
                        break
                    self.player.play_chunk(audio_chunk)

            self.player.mark_stream_end()
            full_reply = " ".join(full_reply_parts)
            self.session_manager.add_assistant_message(full_reply)
            await self.event_bus.publish(EventType.AI_RESPONSE_COMPLETE, response=full_reply)
            logger.info("✅ Response complete.")

        except asyncio.CancelledError:
            logger.info("AI response processing cancelled (interrupted).")
        except Exception as e:
            logger.error("Error in AI turn pipeline: %s", e, exc_info=True)
            await self.event_bus.publish(EventType.SYSTEM_ERROR, error=str(e))
        finally:
            self._is_ai_generating = False
