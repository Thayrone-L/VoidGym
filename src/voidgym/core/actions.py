"""Vocabulário de intenções compartilhado entre jogos 2D."""

from dataclasses import dataclass
from enum import Enum


class Duration(str, Enum):
    """Duração semântica de uma intenção de botão."""

    INSTANT = "instant"
    SHORT = "short"
    HOLD = "hold"
    MAX = "max"
    CHANNEL = "channel"


@dataclass(frozen=True)
class UniversalAction:
    """Intenção de jogador, sem referência a um jogo específico."""

    name: str
    move: int
    aim: str
    jump: Duration | None
    attack: bool
    dash: bool
    special: str | None
    interact: bool
    menu: str | None
    game_time_ms: int


def _action(
    name: str,
    *,
    move: int = 0,
    aim: str = "none",
    jump: Duration | None = None,
    attack: bool = False,
    dash: bool = False,
    special: str | None = None,
    interact: bool = False,
    menu: str | None = None,
    duration: int = 80,
) -> UniversalAction:
    return UniversalAction(name, move, aim, jump, attack, dash, special, interact, menu, duration)


_ROWS = (
    _action("NOOP"),
    _action("LEFT", move=-1),
    _action("RIGHT", move=1),
    _action("JUMP_TAP", jump=Duration.SHORT, duration=50),
    _action("JUMP_HOLD", jump=Duration.HOLD, duration=160),
    _action("LEFT+JUMP_TAP", move=-1, jump=Duration.SHORT, duration=50),
    _action("LEFT+JUMP_HOLD", move=-1, jump=Duration.HOLD, duration=160),
    _action("RIGHT+JUMP_TAP", move=1, jump=Duration.SHORT, duration=50),
    _action("RIGHT+JUMP_HOLD", move=1, jump=Duration.HOLD, duration=160),
    _action("JUMP_MAX", jump=Duration.MAX, duration=450),
    _action("LEFT+JUMP_MAX", move=-1, jump=Duration.MAX, duration=450),
    _action("RIGHT+JUMP_MAX", move=1, jump=Duration.MAX, duration=450),
    _action("ATTACK", attack=True, duration=50),
    _action("LEFT+ATTACK", move=-1, attack=True, duration=50),
    _action("RIGHT+ATTACK", move=1, attack=True, duration=50),
    _action("UP+ATTACK", aim="up", attack=True, duration=50),
    _action("DOWN+ATTACK", aim="down", attack=True, duration=50),
    _action("JUMP+ATTACK", jump=Duration.SHORT, attack=True),
    _action("LEFT+JUMP+ATTACK", move=-1, jump=Duration.SHORT, attack=True),
    _action("RIGHT+JUMP+ATTACK", move=1, jump=Duration.SHORT, attack=True),
    _action("CAST", special="cast", duration=50),
    _action("UP+CAST", aim="up", special="cast", duration=50),
    _action("DOWN+CAST", aim="down", special="cast", duration=50),
    _action("FOCUS_HOLD", special="focus", duration=1500),
    _action("NAIL_ART_CHARGE", special="charge", duration=1500),
    _action("LEFT+NAIL_ART_CHARGE", move=-1, special="charge", duration=1500),
    _action("RIGHT+NAIL_ART_CHARGE", move=1, special="charge", duration=1500),
    _action("UP+NAIL_ART_CHARGE", aim="up", special="charge", duration=1500),
    _action("DOWN+NAIL_ART_CHARGE", aim="down", special="charge", duration=1500),
    _action("DASH", dash=True, duration=50),
    _action("LEFT+DASH", move=-1, dash=True, duration=50),
    _action("RIGHT+DASH", move=1, dash=True, duration=50),
    _action("DOWN+DASH", aim="down", dash=True, duration=50),
    _action("SUPER_DASH_HOLD", special="super_dash", duration=1500),
    _action("DREAM_NAIL", special="dream", duration=160),
    _action("INTERACT", interact=True, aim="up", duration=50),
    _action("MENU_UP", menu="up", duration=50),
    _action("MENU_DOWN", menu="down", duration=50),
    _action("MENU_LEFT", menu="left", duration=50),
    _action("MENU_RIGHT", menu="right", duration=50),
    _action("MENU_SUBMIT", menu="submit", duration=50),
    _action("MENU_CANCEL", menu="cancel", duration=50),
    _action("OPEN_CHARMS", menu="open_charms", duration=50),
)

UNIVERSAL_ACTIONS: tuple[UniversalAction, ...] = _ROWS
UNIVERSAL_ACTION_BY_NAME = {action.name: action for action in UNIVERSAL_ACTIONS}
