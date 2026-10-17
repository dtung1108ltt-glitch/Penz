"""Asynchronous Event Bus for decoupled pipeline signalling and lifecycle management."""
import asyncio
import inspect
import logging
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Union

logger = logging.getLogger(__name__)


class EventType(Enum):
    """System-wide lifecycle and operational events."""
    USER_SPEAKING_START = auto()
    USER_SPEAKING_STOP = auto()
    STT_TRANSCRIBED = auto()
    AI_THINKING_START = auto()
    AI_RESPONSE_CHUNK = auto()
    AI_RESPONSE_COMPLETE = auto()
    AI_SPEAKING_START = auto()
    AI_SPEAKING_STOP = auto()
    INTERRUPT = auto()
    SYSTEM_ERROR = auto()


EventHandler = Union[Callable[..., Any], Callable[..., Coroutine[Any, Any, Any]]]


class EventBus:
    """Pub/Sub event bus supporting both async and sync listener subscriptions."""

    def __init__(self):
        self._subscribers: dict[EventType, list[EventHandler]] = {evt: [] for evt in EventType}

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Register a handler for a specific event."""
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)
            logger.debug("Subscribed %s to event %s", getattr(handler, "__name__", str(handler)), event_type.name)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Unregister a handler."""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)

    async def publish(self, event_type: EventType, **kwargs) -> None:
        """Publish an event asynchronously to all registered listeners."""
        handlers = self._subscribers.get(event_type, [])
        if not handlers:
            return

        tasks = []
        for handler in handlers:
            try:
                if inspect.iscoroutinefunction(handler):
                    tasks.append(asyncio.create_task(handler(**kwargs)))
                else:
                    handler(**kwargs)
            except Exception as e:
                logger.error("Error executing sync handler for event %s: %s", event_type.name, e)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    logger.error("Error executing async handler for event %s: %s", event_type.name, res)
