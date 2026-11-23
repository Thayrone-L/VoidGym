from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from voidgym.training.runtime_monitor import campaign_process_snapshot, _process_role


def test_process_roles_cover_hk_learner_supervisor_actors_and_games() -> None:
    assert _process_role(10, 10, 20, "python.exe", "run_learner.py") == "learner"
    assert _process_role(20, 10, 20, "python.exe", "supervise_hk_learner.py") == "supervisor"
    assert _process_role(30, 10, 20, "python.exe", "run_actor.py") == "actor"
    assert _process_role(30, 30, 20, "python.exe", "run_actor.py") == "actor"
    assert _process_role(40, 10, 20, r"C:\HK\Hollow Knight.exe", "") == "game"


def test_campaign_process_snapshot_reports_memory_and_loaded_modules(
    monkeypatch,
) -> None:
    class FakeProcess:
        def __init__(self, pid: int, executable: str, command: str,
                     parent: FakeProcess | None = None, children: list[FakeProcess] | None = None,
                     modules: list[str] | None = None) -> None:
            self.pid = pid
            self._executable = executable
            self._command = command
            self._parent = parent
            self._children = children or []
            self._modules = modules or []

        def parent(self) -> FakeProcess | None:
            return self._parent

        def children(self, recursive: bool = False) -> list[FakeProcess]:
            del recursive
            return self._children

        def exe(self) -> str:
            return self._executable

        def cmdline(self) -> list[str]:
            return self._command.split()

        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=self.pid * 100)

        def memory_full_info(self) -> SimpleNamespace:
            return SimpleNamespace(uss=self.pid * 10)

        def memory_maps(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(path=path) for path in self._modules]

    learner_pid = os.getpid()
    owner = FakeProcess(100, "panel.exe", "panel")
    supervisor = FakeProcess(learner_pid + 1, r"C:\Users\thayr\Python312\python.exe",
                             r"C:\VoidGym\.venv-cuda\Scripts\python.exe C:\VoidGym\tools\v2\supervise_hk_learner.py", owner)
    learner = FakeProcess(learner_pid, r"C:\Users\thayr\Python312\python.exe",
                          r"C:\VoidGym\.venv-cuda\Scripts\python.exe C:\VoidGym\tools\v2\run_learner.py", supervisor,
                          modules=[r"C:\VoidGym\.venv-cuda\torch\lib\torch_cuda.dll"])
    supervisor._children = [learner]
    actor = FakeProcess(learner_pid + 2, r"C:\Users\thayr\Python312\python.exe",
                        r"C:\VoidGym\.venv-cpu\Scripts\python.exe C:\VoidGym\tools\v2\run_actor.py --device cpu --shared-root E:\VoidGymRuntime\shared\smoke",
                        owner, modules=[r"C:\VoidGym\.venv-cpu\torch\lib\torch_cpu.dll"])
    game = FakeProcess(learner_pid + 3, r"C:\HK\Hollow Knight.exe", "Hollow Knight.exe",
                       actor, modules=[r"C:\HK\UnityPlayer.dll"])
    actor._children = [game]
    owner._children = [supervisor, learner]
    processes = {learner_pid: learner}
    fake_psutil = ModuleType("psutil")
    fake_psutil.Error = OSError  # type: ignore[attr-defined]
    fake_psutil.Process = lambda _pid: processes[learner_pid]  # type: ignore[attr-defined]
    fake_psutil.process_iter = lambda: [actor]  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    records = campaign_process_snapshot(Path(r"E:\VoidGymRuntime\shared\smoke"))

    assert {row["role"] for row in records} == {"learner", "supervisor", "actor", "game"}
    actor_record = next(row for row in records if row["role"] == "actor")
    assert actor_record["executable"].endswith(r".venv-cpu\Scripts\python.exe")
    assert actor_record["image_path"].endswith(r"Python312\python.exe")
    assert actor_record["rss_bytes"] == actor.pid * 100
    assert actor_record["private_bytes"] == actor.pid * 10
    assert actor_record["torch_loaded"] is True
    assert actor_record["cuda_loaded"] is False
