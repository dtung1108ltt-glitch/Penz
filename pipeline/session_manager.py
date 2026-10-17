"""Session and short-term conversation context management."""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ChatTurn:
    """Single conversational turn."""
    role: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class SessionManager:
    """Manages short-term conversation context buffer with sliding window pruning."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history: list[ChatTurn] = []

    def add_user_message(self, content: str) -> None:
        """Append user input to conversation history."""
        clean_text = content.strip()
        if clean_text:
            self.history.append(ChatTurn(role="user", content=clean_text))
            self._prune()

    def add_assistant_message(self, content: str) -> None:
        """Append assistant response to conversation history."""
        clean_text = content.strip()
        if clean_text:
            self.history.append(ChatTurn(role="assistant", content=clean_text))
            self._prune()

    def _prune(self) -> None:
        """Retain only the most recent N turns to prevent context blowup."""
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def get_history(self) -> list[dict[str, str]]:
        """Return history formatted as chat messages list."""
        return [{"role": turn.role, "content": turn.content} for turn in self.history]

    def clear(self) -> None:
        """Reset conversation context."""
        self.history.clear()
        logger.info("Session context cleared.")

    def __len__(self) -> int:
        return len(self.history)

    def __bool__(self) -> bool:
        return True
