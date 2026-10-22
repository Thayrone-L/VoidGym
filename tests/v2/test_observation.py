import numpy as np

from voidgym.core.observation import fit_to_universal, split_observation


def test_letterbox_supported_aspect_ratios() -> None:
    for height, width in ((90, 160), (120, 160), (90, 210)):
        frame = np.full((height, width, 3), 127, dtype=np.uint8)
        fitted = fit_to_universal(frame)
        assert fitted.shape == (108, 192, 3)
        assert fitted.dtype == np.uint8
        assert np.any(np.all(fitted == 0, axis=2)) or (height, width) == (90, 160)


def test_split_observation_is_lossless() -> None:
    original = {
        "pixels": np.zeros((4, 90, 160, 3), dtype=np.uint8),
        "action_history": np.array([1, 2, 3], dtype=np.int64),
        "game_specific": np.array([4], dtype=np.float32),
    }
    universal, specific = split_observation(original)
    rebuilt = {**universal, **specific}
    assert rebuilt.keys() == original.keys()
    assert all(np.array_equal(rebuilt[key], value) for key, value in original.items())
    assert universal.keys() == {"pixels", "action_history"}
    assert specific.keys() == {"game_specific"}
