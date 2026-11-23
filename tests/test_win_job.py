from __future__ import annotations

import ctypes
from types import SimpleNamespace

from hk_agent import win_job


def test_job_uses_kill_on_close_limit(monkeypatch) -> None:
    observed = {}

    def create(*_args):
        return 77

    def set_information(_handle, _kind, info_ptr, _size):
        info = ctypes.cast(info_ptr, ctypes.POINTER(win_job._ExtendedLimit)).contents
        observed["flags"] = info.BasicLimitInformation.LimitFlags
        return 1

    kernel = SimpleNamespace(CreateJobObjectW=create, SetInformationJobObject=set_information, CloseHandle=lambda _handle: 1)
    monkeypatch.setattr(win_job.os, "name", "nt")
    monkeypatch.setattr(win_job.ctypes, "windll", SimpleNamespace(kernel32=kernel), raising=False)
    job = win_job.WindowsJob.create()
    assert observed["flags"] == win_job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    assert job.handle == 77


def test_job_process_ids_and_terminate(monkeypatch) -> None:
    observed = {}

    def query(_handle, kind, info_ptr, _size, _returned):
        assert kind == win_job.JOB_OBJECT_BASIC_PROCESS_ID_LIST
        info = ctypes.cast(info_ptr, ctypes.POINTER(win_job._ProcessIdList)).contents
        info.NumberOfProcessIdsInList = 2
        info.ProcessIdList[0] = 101
        info.ProcessIdList[1] = 202
        return 1

    kernel = SimpleNamespace(
        QueryInformationJobObject=query,
        TerminateJobObject=lambda _handle, code: observed.setdefault("exit_code", code) or 1,
    )
    monkeypatch.setattr(win_job.os, "name", "nt")
    monkeypatch.setattr(win_job.ctypes, "windll", SimpleNamespace(kernel32=kernel), raising=False)
    job = win_job.WindowsJob(55)
    assert job.process_ids() == [101, 202]
    job.terminate(9)
    assert observed["exit_code"] == 9
