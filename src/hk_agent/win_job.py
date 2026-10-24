"""Windows Job Object ownership for runner, worker, and game descendants.

On non-Windows systems the context manager is an intentional no-op.
"""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from typing import Any

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
PROCESS_ALL_ACCESS = 0x1F0FFF
PROCESS_SET_QUOTA = 0x0100
PROCESS_TERMINATE = 0x0001
JOB_OBJECT_BASIC_PROCESS_ID_LIST = 3
PROCESS_SUSPEND_RESUME = 0x0800


class _ProcessIdList(ctypes.Structure):
    _fields_ = [
        ("NumberOfAssignedProcesses", wintypes.DWORD),
        ("NumberOfProcessIdsInList", wintypes.DWORD),
        ("ProcessIdList", ctypes.c_size_t * 1024),
    ]


class _BasicLimit(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong), ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t), ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD), ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimit), ("IoInfo", _IoCounters), ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class WindowsJob:
    def __init__(self, handle: Any = None) -> None:
        self.handle = handle

    @classmethod
    def create(cls) -> WindowsJob:
        if os.name != "nt":
            return cls()
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise ctypes.WinError()
        info = _ExtendedLimit()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(handle, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)):
            error = ctypes.WinError()
            kernel32.CloseHandle(handle)
            raise error
        return cls(handle)

    def assign_pid(self, pid: int) -> None:
        if self.handle is None:
            return
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        process = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, pid)
        if not process:
            raise ctypes.WinError()
        try:
            if not kernel32.AssignProcessToJobObject(self.handle, process):
                raise ctypes.WinError()
        finally:
            kernel32.CloseHandle(process)

    def process_ids(self) -> list[int]:
        """Return current job members; an empty list is the drained state."""
        if self.handle is None:
            return []
        kernel32 = ctypes.windll.kernel32
        query = kernel32.QueryInformationJobObject
        query.restype = wintypes.BOOL
        info = _ProcessIdList()
        if not query(
            self.handle,
            JOB_OBJECT_BASIC_PROCESS_ID_LIST,
            ctypes.byref(info),
            ctypes.sizeof(info),
            None,
        ):
            raise ctypes.WinError()
        return [int(info.ProcessIdList[index]) for index in range(info.NumberOfProcessIdsInList)]

    def terminate(self, exit_code: int = 1) -> None:
        if self.handle is None:
            return
        terminate = ctypes.windll.kernel32.TerminateJobObject
        terminate.restype = wintypes.BOOL
        if not terminate(self.handle, exit_code):
            raise ctypes.WinError()

    def close(self) -> None:
        if self.handle is not None:
            ctypes.windll.kernel32.CloseHandle(self.handle)
            self.handle = None


def resume_process(pid: int) -> None:
    """Resume a process created with CREATE_SUSPENDED."""
    if os.name != "nt":
        return
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.restype = wintypes.BOOL
    process = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
    if not process:
        raise ctypes.WinError()
    try:
        resume = ctypes.windll.ntdll.NtResumeProcess
        resume.restype = wintypes.LONG
        status = resume(process)
        if status < 0:
            raise OSError(f"NtResumeProcess failed with NTSTATUS 0x{status & 0xFFFFFFFF:08X}")
    finally:
        kernel32.CloseHandle(process)


def process_identity(pid: int) -> dict[str, object]:
    """Return stable process identity fields used to validate a persisted launch."""
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
        query = 0x1000
        handle = kernel32.OpenProcess(query, False, pid)
        if not handle:
            raise OSError(f"cannot open process {pid}")
        try:
            creation = wintypes.FILETIME()
            exit_time = wintypes.FILETIME()
            kernel = wintypes.FILETIME()
            user = wintypes.FILETIME()
            if not kernel32.GetProcessTimes(handle, ctypes.byref(creation), ctypes.byref(exit_time), ctypes.byref(kernel), ctypes.byref(user)):
                raise ctypes.WinError()
            image = ctypes.create_unicode_buffer(32768)
            size = wintypes.DWORD(len(image))
            if not kernel32.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(size)):
                raise ctypes.WinError()
            ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
            result: dict[str, object] = {"pid": pid, "creation_time": ticks, "executable": image.value}
            try:
                import psutil  # type: ignore[import-untyped]

                result["parent_pid"] = psutil.Process(pid).ppid()
            except (ImportError, OSError, ValueError):
                pass
            return result
        finally:
            kernel32.CloseHandle(handle)
    try:
        import psutil

        process = psutil.Process(pid)
        return {"pid": pid, "creation_time": process.create_time(), "executable": process.exe(), "parent_pid": process.ppid()}
    except (ImportError, OSError, ValueError) as error:
        raise OSError(f"cannot inspect process {pid}: {error}") from error
