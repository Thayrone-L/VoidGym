from __future__ import annotations

import pytest

from hk_agent.launch_options import windowed_game_arguments


def test_windowed_game_arguments_are_explicit() -> None:
    assert windowed_game_arguments(960, 540) == (
        "-screen-fullscreen",
        "0",
        "-screen-width",
        "960",
        "-screen-height",
        "540",
    )


@pytest.mark.parametrize(("width", "height"), [(319, 540), (960, 319), (7681, 540)])
def test_windowed_game_arguments_reject_unsafe_dimensions(width: int, height: int) -> None:
    with pytest.raises(ValueError):
        windowed_game_arguments(width, height)
