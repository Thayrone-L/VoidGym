from __future__ import annotations


def windowed_game_arguments(width: int, height: int, *, log_file: str | None = None) -> tuple[str, ...]:
    """Return Unity standalone arguments for a deterministic windowed player."""
    for value, name in ((width, "window width"), (height, "window height")):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if not 320 <= value <= 7680:
            raise ValueError(f"{name} must be in [320, 7680]")
    arguments = (
        "-screen-fullscreen",
        "0",
        "-screen-width",
        str(width),
        "-screen-height",
        str(height),
    )
    return arguments + (("-logFile", log_file) if log_file is not None else ())
