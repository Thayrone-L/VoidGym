from __future__ import annotations

import ctypes
from types import SimpleNamespace

import hk_agent.window_placement as placement


def test_slot_positions_and_primary_monitor_fallback() -> None:
    assert placement.slot_position(0) == (1920, 0, 960, 540)
    assert placement.slot_position(1) == (2880, 0, 960, 540)
    assert placement.slot_position(0, second_monitor=False) == (0, 0, 960, 540)
    assert placement.slot_position(1, second_monitor=False) == (960, 0, 960, 540)


def test_window_search_filters_by_pid(monkeypatch) -> None:
    def enum_windows(callback, _param):
        callback(11, 0)
        callback(22, 0)

    def get_pid(hwnd, pointer):
        ctypes.cast(pointer, ctypes.POINTER(placement.wintypes.DWORD)).contents.value = 101 if hwnd == 11 else 202
        return 1

    user32 = SimpleNamespace(
        EnumWindows=enum_windows, GetWindowThreadProcessId=get_pid,
        IsWindowVisible=lambda _hwnd: 1,
    )
    monkeypatch.setattr(placement.os, "name", "nt")
    monkeypatch.setattr(placement.ctypes, "windll", SimpleNamespace(user32=user32), raising=False)
    monkeypatch.setattr(placement.ctypes, "WINFUNCTYPE", ctypes.CFUNCTYPE, raising=False)
    assert placement.find_window_for_pid(202) == 22


def test_place_window_moves_without_resizing_and_steps_by_outer_width(monkeypatch) -> None:
    calls: list[tuple[int, int, int, int, int, int, int]] = []

    def get_window_rect(_hwnd, pointer):
        rect = ctypes.cast(pointer, ctypes.POINTER(placement.wintypes.RECT)).contents
        rect.left, rect.top, rect.right, rect.bottom = 100, 100, 1076, 679  # 976x579 outer
        return 1

    def set_window_pos(hwnd, after, x, y, cx, cy, flags):
        calls.append((hwnd, after, x, y, cx, cy, flags))
        return 1

    user32 = SimpleNamespace(
        GetWindowRect=get_window_rect,
        SetWindowPos=set_window_pos,
        GetSystemMetrics=lambda _metric: 2,
    )
    monkeypatch.setattr(placement.os, "name", "nt")
    monkeypatch.setattr(placement.ctypes, "windll", SimpleNamespace(user32=user32), raising=False)
    assert placement.place_window(55, 1) is True
    hwnd, _after, x, y, _cx, _cy, flags = calls[0]
    assert hwnd == 55
    assert (x, y) == (1920 + 976, 0)
    assert flags & placement.SWP_NOSIZE, "placement must never resize the client area"


def test_slot_origin_validates_inputs() -> None:
    assert placement.slot_origin(0, 976, second_monitor=True) == (1920, 0)
    assert placement.slot_origin(1, 976, second_monitor=False) == (976, 0)
