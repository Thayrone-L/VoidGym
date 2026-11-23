"""Place and find only the game window owned by a known process ID."""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes


def slot_position(slot_id: int, *, second_monitor: bool = True) -> tuple[int, int, int, int]:
    if slot_id not in (0, 1):
        raise ValueError("slot_id must be 0 or 1")
    if second_monitor:
        return ((1920 if slot_id == 0 else 2880), 0, 960, 540)
    return ((0 if slot_id == 0 else 960), 0, 960, 540)


def find_window_for_pid(pid: int) -> int | None:
    if os.name != "nt":
        return None
    user32 = ctypes.windll.user32
    found: list[int] = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def visit(hwnd: int, _: int) -> bool:
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
            return False
        return True

    callback = callback_type(visit)
    user32.EnumWindows(callback, 0)
    return found[0] if found else None


SWP_NOSIZE = 0x0001
SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
# Move only. Resizing the outer window shrinks Unity's client area (e.g. 960x540
# became 944x501), which changes the captured frame size mid-run and breaks the
# menu readiness reference. The client size belongs to -screen-width/-height.
PLACEMENT_FLAGS = SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_SHOWWINDOW


def slot_origin(slot_id: int, outer_width: int, *, second_monitor: bool) -> tuple[int, int]:
    """Top-left corner for a slot, stepping by the window's real outer width."""
    if slot_id not in (0, 1):
        raise ValueError("slot_id must be 0 or 1")
    if outer_width <= 0:
        raise ValueError("outer_width must be positive")
    base_x = slot_position(0, second_monitor=second_monitor)[0]
    return base_x + slot_id * outer_width, 0


def place_window(hwnd: int, slot_id: int) -> bool:
    """Move an already identified window; returns False when the move fails."""
    if os.name != "nt" or not hwnd:
        return False
    user32 = ctypes.windll.user32
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return False
    outer_width = rect.right - rect.left
    if outer_width <= 0:
        return False
    second_monitor_exists = user32.GetSystemMetrics(MONITOR_COUNT_METRIC) > 1
    x, y = slot_origin(slot_id, outer_width, second_monitor=second_monitor_exists)
    return bool(user32.SetWindowPos(hwnd, 0, x, y, 0, 0, PLACEMENT_FLAGS))


def place_window_for_pid(pid: int, slot_id: int) -> bool:
    hwnd = find_window_for_pid(pid)
    if hwnd is None:
        return False
    return place_window(hwnd, slot_id)


MONITOR_COUNT_METRIC = 80
