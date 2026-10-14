from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class SessionError(RuntimeError):
    """Raised when a session command is invalid for the current state."""


class SessionState(Enum):
    CREATED = auto()
    CONNECTING = auto()
    HANDSHAKING = auto()
    READY = auto()
    RESETTING = auto()
    STEPPING = auto()
    FAULTED = auto()
    STOPPING = auto()
    STOPPED = auto()


_ALLOWED: dict[SessionState, frozenset[SessionState]] = {
    SessionState.CREATED: frozenset({SessionState.CONNECTING, SessionState.STOPPING}),
    SessionState.CONNECTING: frozenset({SessionState.HANDSHAKING, SessionState.FAULTED}),
    SessionState.HANDSHAKING: frozenset({SessionState.READY, SessionState.FAULTED}),
    SessionState.READY: frozenset(
        {SessionState.RESETTING, SessionState.STEPPING, SessionState.FAULTED, SessionState.STOPPING}
    ),
    SessionState.RESETTING: frozenset({SessionState.READY, SessionState.FAULTED}),
    SessionState.STEPPING: frozenset({SessionState.READY, SessionState.FAULTED}),
    SessionState.FAULTED: frozenset({SessionState.STOPPING}),
    SessionState.STOPPING: frozenset({SessionState.STOPPED}),
    SessionState.STOPPED: frozenset(),
}


@dataclass(slots=True)
class SessionMachine:
    state: SessionState = SessionState.CREATED
    last_inbound_seq: int = -1

    def transition(self, target: SessionState) -> None:
        if target not in _ALLOWED[self.state]:
            raise SessionError(f"Invalid state transition: {self.state.name} -> {target.name}")
        self.state = target

    def accept_sequence(self, sequence: int) -> None:
        if sequence <= self.last_inbound_seq:
            raise SessionError(f"Sequence {sequence} is not newer than {self.last_inbound_seq}")
        self.last_inbound_seq = sequence
