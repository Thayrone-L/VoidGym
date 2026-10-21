"""Conversão explícita entre tempo de jogo e tempo de parede."""


def _validate_scale(time_scale: float) -> None:
    if not 0.25 <= time_scale <= 4.0:
        raise ValueError("time_scale deve estar entre 0.25 e 4.0")


def game_ms_to_real_ms(game_ms: int, time_scale: float) -> float:
    _validate_scale(time_scale)
    if game_ms < 0:
        raise ValueError("game_ms não pode ser negativo")
    return game_ms / time_scale


def real_ms_to_game_ms(real_ms: float, time_scale: float) -> float:
    _validate_scale(time_scale)
    if real_ms < 0:
        raise ValueError("real_ms não pode ser negativo")
    return real_ms * time_scale
