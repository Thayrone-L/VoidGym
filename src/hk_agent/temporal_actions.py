from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from hk_agent.gym_env import BUTTON_ORDER

DurationKind = Literal["base", "tap", "hold", "focus", "charge"]


@dataclass(frozen=True, slots=True)
class TemporalActionTemplate:
    """A policy action whose buttons and bounded duration are both explicit."""

    name: str
    buttons: tuple[str, ...]
    duration: DurationKind = "base"

    @property
    def raw_action_index(self) -> int:
        return sum(1 << BUTTON_ORDER.index(button) for button in self.buttons)

    def resolve_lease_ms(self, base_lease_ms: int) -> int:
        if self.duration == "tap":
            return min(base_lease_ms, 50)
        if self.duration == "hold":
            return min(250, max(base_lease_ms * 2, 120))
        if self.duration in {"focus", "charge"}:
            # Focus and Nail Arts both need an uninterrupted long press. The
            # automatic release executes the charged action and the backend
            # watchdog still enforces a short upper bound.
            return 1_500
        return base_lease_ms


TEMPORAL_COMBAT_ACTIONS: tuple[TemporalActionTemplate, ...] = (
    TemporalActionTemplate("NOOP", ()),
    TemporalActionTemplate("LEFT", ("LEFT",)),
    TemporalActionTemplate("RIGHT", ("RIGHT",)),
    TemporalActionTemplate("JUMP_TAP", ("JUMP",), "tap"),
    TemporalActionTemplate("JUMP_HOLD", ("JUMP",), "hold"),
    TemporalActionTemplate("LEFT+JUMP_TAP", ("LEFT", "JUMP"), "tap"),
    TemporalActionTemplate("LEFT+JUMP_HOLD", ("LEFT", "JUMP"), "hold"),
    TemporalActionTemplate("RIGHT+JUMP_TAP", ("RIGHT", "JUMP"), "tap"),
    TemporalActionTemplate("RIGHT+JUMP_HOLD", ("RIGHT", "JUMP"), "hold"),
    TemporalActionTemplate("ATTACK", ("ATTACK",), "tap"),
    TemporalActionTemplate("LEFT+ATTACK", ("LEFT", "ATTACK"), "tap"),
    TemporalActionTemplate("RIGHT+ATTACK", ("RIGHT", "ATTACK"), "tap"),
    TemporalActionTemplate("UP+ATTACK", ("UP", "ATTACK"), "tap"),
    TemporalActionTemplate("DOWN+ATTACK", ("DOWN", "ATTACK"), "tap"),
    TemporalActionTemplate("JUMP+ATTACK", ("JUMP", "ATTACK"), "base"),
    TemporalActionTemplate(
        "LEFT+JUMP+ATTACK", ("LEFT", "JUMP", "ATTACK"), "base"
    ),
    TemporalActionTemplate(
        "RIGHT+JUMP+ATTACK", ("RIGHT", "JUMP", "ATTACK"), "base"
    ),
    TemporalActionTemplate("CAST", ("CAST",), "tap"),
    TemporalActionTemplate("UP+CAST", ("UP", "CAST"), "tap"),
    TemporalActionTemplate("DOWN+CAST", ("DOWN", "CAST"), "tap"),
    TemporalActionTemplate("FOCUS_HOLD", ("CAST",), "focus"),
    TemporalActionTemplate("NAIL_ART_CHARGE", ("ATTACK",), "charge"),
    TemporalActionTemplate(
        "LEFT+NAIL_ART_CHARGE", ("LEFT", "ATTACK"), "charge"
    ),
    TemporalActionTemplate(
        "RIGHT+NAIL_ART_CHARGE", ("RIGHT", "ATTACK"), "charge"
    ),
    TemporalActionTemplate("UP+NAIL_ART_CHARGE", ("UP", "ATTACK"), "charge"),
    TemporalActionTemplate(
        "DOWN+NAIL_ART_CHARGE", ("DOWN", "ATTACK"), "charge"
    ),
    TemporalActionTemplate("DASH", ("DASH",), "tap"),
    TemporalActionTemplate("LEFT+DASH", ("LEFT", "DASH"), "tap"),
    TemporalActionTemplate("RIGHT+DASH", ("RIGHT", "DASH"), "tap"),
    TemporalActionTemplate("DOWN+DASH", ("DOWN", "DASH"), "tap"),
    TemporalActionTemplate("SUPER_DASH_HOLD", ("SUPER_DASH",), "charge"),
    TemporalActionTemplate("DREAM_NAIL", ("DREAM_NAIL",), "hold"),
)

PLAYER_INTERACTION_ACTIONS: tuple[TemporalActionTemplate, ...] = (
    TemporalActionTemplate("INTERACT", ("UP",), "tap"),
    TemporalActionTemplate("MENU_UP", ("UP",), "tap"),
    TemporalActionTemplate("MENU_DOWN", ("DOWN",), "tap"),
    TemporalActionTemplate("MENU_LEFT", ("LEFT",), "tap"),
    TemporalActionTemplate("MENU_RIGHT", ("RIGHT",), "tap"),
    TemporalActionTemplate("MENU_SUBMIT", ("JUMP",), "tap"),
    TemporalActionTemplate("MENU_CANCEL", ("DASH",), "tap"),
)


def temporal_observation_space(
    source: gym.Space[dict[str, NDArray[np.generic]]], *, number_of_actions: int
) -> gym.spaces.Dict:
    source_dict = source
    if not isinstance(source_dict, gym.spaces.Dict):
        raise TypeError("temporal actions require a Dict observation space")
    history = source_dict.spaces.get("action_history")
    if history is None or len(history.shape) != 1:
        raise TypeError("temporal actions require a one-dimensional action_history")
    return gym.spaces.Dict(
        {
            **source_dict.spaces,
            "action_history": gym.spaces.MultiDiscrete(
                np.full(history.shape[0], number_of_actions, dtype=np.int64)
            ),
        }
    )


class TemporalActionHistory:
    """Replace ambiguous raw-button history with policy action tokens."""

    def __init__(self, length: int) -> None:
        self._values = [0] * length

    def reset(self) -> None:
        self._values = [0] * len(self._values)

    def append(self, action: int) -> None:
        if self._values:
            self._values = (self._values + [action])[-len(self._values) :]

    def apply(
        self, observation: dict[str, NDArray[np.generic]]
    ) -> dict[str, NDArray[np.generic]]:
        result = dict(observation)
        result["action_history"] = np.asarray(self._values, dtype=np.int64)
        return result


def action_contract(
    actions: tuple[TemporalActionTemplate, ...], *, base_lease_ms: int
) -> list[dict[str, object]]:
    return [
        {
            "name": action.name,
            "buttons": list(action.buttons),
            "raw_action_index": action.raw_action_index,
            "duration": action.duration,
            "lease_ms": action.resolve_lease_ms(base_lease_ms),
            "release_after": True,
        }
        for action in actions
    ]
