from typing import Any, Callable, Dict, List, Literal
from dataclasses import dataclass
import time


type EventType = Literal['brain_created', 'brain_stats_updated', 'neuron_fired']


@dataclass(slots=True)
class Event:
    """Base event carrying a type, payload, and timestamp."""
    event_type: EventType
    data: Dict[str, Any]
    timestamp: float

    def __repr__(self) -> str:
        return f"Event(type={self.event_type!r}, data={self.data!r}, ts={self.timestamp:.6f})"


class EventBus:
    """Simple synchronous event bus.

    - Subscribe handlers per event type
    - Unsubscribe handlers
    - Publish existing Event objects
    - Emit convenience method to build and send an Event
    """

    def __init__(self) -> None:
        self._subscribers: Dict[EventType, List[Callable[[Event], None]]] = {}

    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """Register a handler for a specific event type."""
        handlers = self._subscribers.setdefault(event_type, [])
        if handler not in handlers:
            handlers.append(handler)

    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """Remove a previously registered handler."""
        handlers = self._subscribers.get(event_type)
        if not handlers:
            return
        try:
            handlers.remove(handler)
        except ValueError:
            pass
        if not handlers:
            # Clean up empty lists
            del self._subscribers[event_type]

    def publish(self, event: Event) -> None:
        """Publish an Event to all subscribers of its type."""
        for handler in self._subscribers.get(event.event_type, []):
            try:
                handler(event)
            except Exception:
                # Intentionally suppress to avoid breaking dispatch chain
                # Callers can add logging around this if needed
                pass

    def emit(self, event_type: EventType, data: Dict[str, Any], timestamp: float | None = None) -> Event:
        """Create and publish an Event."""
        ts = time.time() if timestamp is None else timestamp
        event = Event(event_type=event_type, data=data, timestamp=ts)
        self.publish(event)
        return event

    # Convenience helpers for known event types
    def broadcast_brain_created(self, data: Dict[str, Any], timestamp: float | None = None) -> Event:
        return self.emit('brain_created', data, timestamp)

    def broadcast_brain_stats_updated(self, data: Dict[str, Any], timestamp: float | None = None) -> Event:
        return self.emit('brain_stats_updated', data, timestamp)

    def broadcast_neuron_fired(self, data: Dict[str, Any], timestamp: float | None = None) -> Event:
        return self.emit('neuron_fired', data, timestamp)
