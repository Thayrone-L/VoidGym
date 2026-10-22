"""Rótulos espaciais aproximados sobre a imagem universal."""

from enum import IntEnum


class Sem(IntEnum):
    """Mapa grosseiro (células de ~4×4 px), com IGNORE sem informação do jogo."""

    EMPTY = 0
    SOLID = 1
    PLATFORM = 2
    HAZARD = 3
    ENEMY = 4
    PLAYER = 5
    BREAKABLE = 6
    PICKUP = 7
    INTERACTIVE = 8
    EXIT = 9


SEM_GRID = (27, 48)
IGNORE = 255
