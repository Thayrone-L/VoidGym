"""Conversões de duração entre relógio de jogo e relógio real."""

from voidgym.core.errors import VoidGymValueError as ValueError


def _scale(value: float) -> float:
    if not 0.25 <= value <= 4.0:
        raise ValueError("time_scale deve estar entre 0.25 e 4.0")
    return value


def game_ms_to_real_ms(game_ms: int, time_scale: float) -> float:
    if game_ms < 0:
        raise ValueError("game_ms não pode ser negativo")
    return game_ms / _scale(time_scale)


def real_ms_to_game_ms(real_ms: int, time_scale: float) -> float:
    if real_ms < 0:
        raise ValueError("real_ms não pode ser negativo")
    return real_ms * _scale(time_scale)
