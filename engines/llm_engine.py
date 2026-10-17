"""Conversational Large Language Model (LLM) inference engines with sentence chunking."""
import asyncio
import logging
import re
from typing import AsyncGenerator, Optional

from config.model_config import LLMConfig
from engines.base_engine import BaseLLMEngine

logger = logging.getLogger(__name__)

# Regular expression to match natural Vietnamese/English sentence boundaries
SENTENCE_SPLIT_REGEX = re.compile(r"([.!?;:\n]+)")


class SentenceChunker:
    """Accumulates streaming tokens and yields natural sentence/clause chunks for TTS.
    
    This dramatically reduces Time-To-First-Byte (TTFB) while maintaining natural intonation.
    """

    def __init__(self, min_chars: int = 12):
        self.min_chars = min_chars
        self.buffer = ""

    def feed(self, token: str) -> list[str]:
        """Feed a new token into the chunker and extract completed sentences."""
        self.buffer += token
        chunks = []

        parts = SENTENCE_SPLIT_REGEX.split(self.buffer)
        if len(parts) > 2:
            # We have at least: [sentence_text, delimiter, remaining_buffer]
            i = 0
            while i < len(parts) - 1:
                sentence_candidate = parts[i] + parts[i + 1]
                if len(sentence_candidate.strip()) >= self.min_chars or i + 2 == len(parts):
                    chunks.append(sentence_candidate.strip())
                    i += 2
                else:
                    # Merge with next
                    if i + 2 < len(parts):
                        parts[i + 2] = sentence_candidate + parts[i + 2]
                    i += 2

            self.buffer = parts[-1] if len(parts) % 2 == 1 else ""

        return chunks

    def flush(self) -> Optional[str]:
        """Flush any remaining text in the buffer upon generation completion."""
        remainder = self.buffer.strip()
        self.buffer = ""
        return remainder if remainder else None


class OllamaLLMEngine(BaseLLMEngine):
    """Low-latency streaming LLM engine using local Ollama instance."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._is_ready = False
        self._client = None

    def initialize(self) -> None:
        try:
            import ollama
            self._client = ollama.AsyncClient(host=self.config.host)
            self._is_ready = True
            logger.info("OllamaLLMEngine initialized with model: %s", self.config.model_name)
        except ImportError:
            logger.error("ollama package not installed. Run: pip install ollama")
            raise

    def shutdown(self) -> None:
        self._client = None
        self._is_ready = False
        logger.info("OllamaLLMEngine shutdown.")

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def _prepare_messages(self, prompt: str, history: Optional[list[dict[str, str]]] = None) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.config.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, history: Optional[list[dict[str, str]]] = None) -> str:
        """Synchronous wrapper for generation."""
        import ollama
        client = ollama.Client(host=self.config.host)
        messages = self._prepare_messages(prompt, history)
        response = client.chat(
            model=self.config.model_name,
            messages=messages,
            options={"temperature": self.config.temperature},
        )
        return response["message"]["content"]

    async def generate_stream(
        self, prompt: str, history: Optional[list[dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream chunks from Ollama split into natural sentence clauses."""
        if not self._is_ready or self._client is None:
            raise RuntimeError("OllamaLLMEngine is not initialized.")

        messages = self._prepare_messages(prompt, history)
        chunker = SentenceChunker()

        response_stream = await self._client.chat(
            model=self.config.model_name,
            messages=messages,
            options={"temperature": self.config.temperature},
            stream=True,
        )

        async for chunk in response_stream:
            token = chunk.get("message", {}).get("content", "")
            if not token:
                continue

            for sentence in chunker.feed(token):
                if sentence:
                    yield sentence

        remainder = chunker.flush()
        if remainder:
            yield remainder


class OpenAILLMEngine(BaseLLMEngine):
    """Streaming LLM engine using OpenAI API or compatible server."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._is_ready = False
        self._client = None

    def initialize(self) -> None:
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.config.api_key)
            self._is_ready = True
            logger.info("OpenAILLMEngine initialized.")
        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
            raise

    def shutdown(self) -> None:
        self._client = None
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def _prepare_messages(self, prompt: str, history: Optional[list[dict[str, str]]] = None) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.config.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, history: Optional[list[dict[str, str]]] = None) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.config.api_key)
        messages = self._prepare_messages(prompt, history)
        response = client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content or ""

    async def generate_stream(
        self, prompt: str, history: Optional[list[dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        if not self._is_ready or self._client is None:
            raise RuntimeError("OpenAILLMEngine is not initialized.")

        messages = self._prepare_messages(prompt, history)
        chunker = SentenceChunker()

        stream = await self._client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )

        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if not token:
                continue
            for sentence in chunker.feed(token):
                if sentence:
                    yield sentence

        remainder = chunker.flush()
        if remainder:
            yield remainder


class LocalMergedLLMEngine(BaseLLMEngine):
    """Local LLM Engine powered by fine-tuned & merged model using Hugging Face Transformers."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.model_path = getattr(self.config, "local_model_path", "models/penz_voice_llm_merged")
        self.model = None
        self.tokenizer = None
        self._is_ready = False

    def initialize(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            logger.info("Loading local merged LLM from: %s", self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                dtype=torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )
            self._is_ready = True
            logger.info("Local merged LLM loaded successfully.")
        except Exception as e:
            logger.error("Failed to load local merged LLM (%s): %s", self.model_path, e)
            raise

    def shutdown(self) -> None:
        self.model = None
        self.tokenizer = None
        self._is_ready = False
        logger.info("LocalMergedLLMEngine shutdown.")

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def _prepare_messages(self, prompt: str, history: Optional[list[dict[str, str]]] = None) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.config.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt: str, history: Optional[list[dict[str, str]]] = None) -> str:
        if not self._is_ready or self.model is None or self.tokenizer is None:
            raise RuntimeError("LocalMergedLLMEngine is not initialized.")

        import torch
        messages = self._prepare_messages(prompt, history)
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = outputs[0][inputs.input_ids.shape[1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    async def generate_stream(
        self, prompt: str, history: Optional[list[dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        if not self._is_ready or self.model is None or self.tokenizer is None:
            raise RuntimeError("LocalMergedLLMEngine is not initialized.")

        from threading import Thread
        from transformers import TextIteratorStreamer
        import torch

        messages = self._prepare_messages(prompt, history)
        prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        chunker = SentenceChunker()
        loop = asyncio.get_running_loop()

        def _get_next_token():
            try:
                return next(streamer)
            except StopIteration:
                return None

        while True:
            token = await loop.run_in_executor(None, _get_next_token)
            if token is None:
                break
            for sentence in chunker.feed(token):
                if sentence:
                    yield sentence

        remainder = chunker.flush()
        if remainder:
            yield remainder
        thread.join()


class MockLLMEngine(BaseLLMEngine):
    """Mock LLM Engine producing instant streaming sentences for tests."""

    def __init__(self, response_text: str = "Chào bạn! Tôi có thể hỗ trợ gì cho bạn hôm nay?"):
        self.response_text = response_text
        self._is_ready = False

    def initialize(self) -> None:
        self._is_ready = True
        logger.info("MockLLMEngine initialized.")

    def shutdown(self) -> None:
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def generate(self, prompt: str, history: Optional[list[dict[str, str]]] = None) -> str:
        return self.response_text

    async def generate_stream(
        self, prompt: str, history: Optional[list[dict[str, str]]] = None
    ) -> AsyncGenerator[str, None]:
        # Simulate slight streaming delay
        sentences = [
            "Chào bạn!",
            "Tôi có thể hỗ trợ gì cho bạn hôm nay?",
        ]
        for sentence in sentences:
            await asyncio.sleep(0.05)
            yield sentence


def get_llm_engine(config: Optional[LLMConfig] = None) -> BaseLLMEngine:
    """Factory function for LLM engines."""
    cfg = config or LLMConfig()
    if cfg.provider in ("local", "merged", "transformers"):
        return LocalMergedLLMEngine(cfg)
    elif cfg.provider == "mock":
        return MockLLMEngine()
    elif cfg.provider == "ollama":
        return OllamaLLMEngine(cfg)
    elif cfg.provider == "openai":
        return OpenAILLMEngine(cfg)
    else:
        logger.warning("Unknown LLM provider '%s', defaulting to MockLLMEngine.", cfg.provider)
        return MockLLMEngine()
