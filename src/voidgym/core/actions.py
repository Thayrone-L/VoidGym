"""Intenções de controle comuns a jogos de plataforma 2D."""

from dataclasses import dataclass
from enum import Enum

from voidgym.core.errors import VoidGymValueError as ValueError


class Duration(str, Enum):
    """Duração semântica de uma intenção, em tempo de jogo."""

    INSTANT = "instant"
    SHORT = "short"
    HOLD = "hold"
    MAX = "max"
    CHANNEL = "channel"


@dataclass(frozen=True, slots=True)
class UniversalAction:
    """Intenção de controle com duração em tempo de jogo.

    ``name`` identifica a intenção; ``move`` dá a direção horizontal
    (-1/0/+1); ``aim`` mira para cima/baixo ou não mira; ``jump`` descreve o
    tipo de pulo; ``attack`` e ``dash`` pedem ataque e esquiva; ``special``
    identifica uma habilidade; ``interact`` interage com o cenário; ``menu``
    navega/confirma/cancela; ``game_time_ms`` é a duração em ms de jogo,
    inclusive para ataques carregados.
    """

    name: str
    move: int = 0
    aim: str = "none"
    jump: Duration | None = None
    attack: bool = False
    dash: bool = False
    special: str | None = None
    interact: bool = False
    menu: str | None = None
    game_time_ms: int = 80

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name deve ser uma string não vazia")
        if isinstance(self.move, bool) or not isinstance(self.move, int) or self.move not in (-1, 0, 1):
            raise ValueError("move deve ser -1, 0 ou 1")
        if self.aim not in {"none", "up", "down"}:
            raise ValueError("aim deve ser none, up ou down")
        if self.jump is not None and not isinstance(self.jump, Duration):
            raise ValueError("jump deve ser Duration ou None")
        for name, value in (("attack", self.attack), ("dash", self.dash), ("interact", self.interact)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} deve ser booleano")
        if self.special not in {None, "cast", "focus", "super_dash", "dream"}:
            raise ValueError("special deve ser cast, focus, super_dash, dream ou None")
        if self.menu not in {None, "up", "down", "left", "right", "submit", "cancel"}:
            raise ValueError("menu deve ser up, down, left, right, submit, cancel ou None")
        if (isinstance(self.game_time_ms, bool) or not isinstance(self.game_time_ms, int)
                or self.game_time_ms < 0):
            raise ValueError("game_time_ms deve ser inteiro não negativo")


def _a(name: str, *, move: int = 0, aim: str = "none", jump: Duration | None = None,
       attack: bool = False, dash: bool = False, special: str | None = None,
       interact: bool = False, menu: str | None = None, ms: int = 80) -> UniversalAction:
    return UniversalAction(name, move, aim, jump, attack, dash, special, interact, menu, ms)


UNIVERSAL_ACTIONS: tuple[UniversalAction, ...] = (
    _a("NOOP"), _a("MOVE_LEFT", move=-1), _a("MOVE_RIGHT", move=1),
    _a("JUMP_SHORT", jump=Duration.SHORT, ms=50), _a("JUMP_HOLD", jump=Duration.HOLD, ms=120),
    _a("MOVE_LEFT+JUMP_SHORT", move=-1, jump=Duration.SHORT, ms=50),
    _a("MOVE_LEFT+JUMP_HOLD", move=-1, jump=Duration.HOLD, ms=120),
    _a("MOVE_RIGHT+JUMP_SHORT", move=1, jump=Duration.SHORT, ms=50),
    _a("MOVE_RIGHT+JUMP_HOLD", move=1, jump=Duration.HOLD, ms=120),
    _a("JUMP_MAX", jump=Duration.MAX, ms=450), _a("MOVE_LEFT+JUMP_MAX", move=-1, jump=Duration.MAX, ms=450),
    _a("MOVE_RIGHT+JUMP_MAX", move=1, jump=Duration.MAX, ms=450),
    _a("ATTACK", attack=True, ms=50), _a("MOVE_LEFT+ATTACK", move=-1, attack=True, ms=50),
    _a("MOVE_RIGHT+ATTACK", move=1, attack=True, ms=50), _a("AIM_UP+ATTACK", aim="up", attack=True, ms=50),
    _a("AIM_DOWN+ATTACK", aim="down", attack=True, ms=50), _a("JUMP+ATTACK", jump=Duration.INSTANT, attack=True),
    _a("MOVE_LEFT+JUMP+ATTACK", move=-1, jump=Duration.INSTANT, attack=True),
    _a("MOVE_RIGHT+JUMP+ATTACK", move=1, jump=Duration.INSTANT, attack=True),
    _a("SPECIAL_CAST", special="cast", ms=50), _a("AIM_UP+CAST", aim="up", special="cast", ms=50),
    _a("AIM_DOWN+CAST", aim="down", special="cast", ms=50), _a("FOCUS", special="focus", ms=1500),
    _a("NAIL_ART", attack=True, ms=1500),
    _a("MOVE_LEFT+NAIL_ART", move=-1, attack=True, ms=1500),
    _a("MOVE_RIGHT+NAIL_ART", move=1, attack=True, ms=1500),
    _a("AIM_UP+NAIL_ART", aim="up", attack=True, ms=1500),
    _a("AIM_DOWN+NAIL_ART", aim="down", attack=True, ms=1500),
    _a("DASH", dash=True, ms=50), _a("MOVE_LEFT+DASH", move=-1, dash=True, ms=50),
    _a("MOVE_RIGHT+DASH", move=1, dash=True, ms=50), _a("AIM_DOWN+DASH", aim="down", dash=True, ms=50),
    _a("SUPER_DASH", special="super_dash", ms=1500), _a("DREAM", special="dream", ms=120),
    _a("INTERACT", interact=True, ms=50), _a("MENU_UP", menu="up", ms=50),
    _a("MENU_DOWN", menu="down", ms=50), _a("MENU_LEFT", menu="left", ms=50),
    _a("MENU_RIGHT", menu="right", ms=50), _a("MENU_SUBMIT", menu="submit", ms=50),
    _a("MENU_CANCEL", menu="cancel", ms=50), _a("OPEN_MENU", ms=50),
)
