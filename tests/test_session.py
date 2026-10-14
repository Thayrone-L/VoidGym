from __future__ import annotations

import unittest

from hk_agent.session import SessionError, SessionMachine, SessionState


class SessionMachineTests(unittest.TestCase):
    def test_happy_path(self) -> None:
        machine = SessionMachine()
        for state in (
            SessionState.CONNECTING,
            SessionState.HANDSHAKING,
            SessionState.READY,
            SessionState.STEPPING,
            SessionState.READY,
            SessionState.STOPPING,
            SessionState.STOPPED,
        ):
            machine.transition(state)
        self.assertEqual(machine.state, SessionState.STOPPED)

    def test_rejects_step_before_handshake(self) -> None:
        machine = SessionMachine()
        with self.assertRaisesRegex(SessionError, "Invalid state transition"):
            machine.transition(SessionState.STEPPING)

    def test_rejects_duplicate_and_old_sequences(self) -> None:
        machine = SessionMachine()
        machine.accept_sequence(5)
        with self.assertRaisesRegex(SessionError, "not newer"):
            machine.accept_sequence(5)
        with self.assertRaisesRegex(SessionError, "not newer"):
            machine.accept_sequence(4)
