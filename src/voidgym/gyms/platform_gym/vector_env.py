"""Spawn-based vector environment whose workers never import stable-baselines3 or torch."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Sequence
from typing import Any, Self, cast

import numpy as np

from voidgym.envs.adapter_env import AdapterEnv
from voidgym.gyms.platform_gym.vec_worker import worker_startup_record


def _worker(remote: Any, parent_remote: Any, env_factory: Callable[[], AdapterEnv],
            worker_index: int) -> None:
    parent_remote.close()
    env: AdapterEnv | None = None
    try:
        env = env_factory()
        remote.send(("ready", worker_startup_record(worker_index), env.actions))
        while True:
            try:
                command, data = remote.recv()
            except EOFError:
                break
            if command == "step":
                observation, reward, terminated, truncated, info = env.step(int(data))
                done = bool(terminated or truncated)
                info["TimeLimit.truncated"] = bool(truncated and not terminated)
                if done:
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()
                    remote.send((observation, reward, done, info, reset_info))
                else:
                    remote.send((observation, reward, done, info, {}))
            elif command == "reset":
                observation, info = env.reset(seed=data[0], options=data[1])
                remote.send((observation, info))
            elif command == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif command == "get_attr":
                remote.send(getattr(env, data))
            elif command == "set_attr":
                setattr(env, data[0], data[1])
                remote.send(None)
            elif command == "close":
                remote.send(None)
                break
            else:
                raise NotImplementedError(f"unsupported vector environment command: {command}")
    finally:
        if env is not None:
            env.close()
        remote.close()


class PlatformSubprocVecEnv:
    """Small SB3-compatible subset used by the standalone platform pretrainer."""

    def __init__(self, env_fns: Sequence[Callable[[], AdapterEnv]], *,
                 start_method: str = "spawn") -> None:
        if not env_fns:
            raise ValueError("at least one platform environment is required")
        context = cast(Any, mp.get_context(start_method))
        self.remotes: list[Any] = []
        self.processes: list[mp.Process] = []
        self._seeds: list[int | None] = [None] * len(env_fns)
        self._options: list[dict[str, Any] | None] = [None] * len(env_fns)
        self.reset_infos: list[dict[str, Any]] = [{} for _ in env_fns]
        self.waiting = False
        self.closed = False
        self.worker_startup: list[dict[str, Any]] = []
        for index, env_fn in enumerate(env_fns):
            remote, work_remote = context.Pipe()
            process = context.Process(
                target=_worker, args=(work_remote, remote, env_fn, index), daemon=True,
                name=f"voidgym-platform-worker-{index}",
            )
            process.start()
            work_remote.close()
            self.remotes.append(remote)
            self.processes.append(process)
        try:
            ready = [remote.recv() for remote in self.remotes]
            if any(item[0] != "ready" for item in ready):
                raise RuntimeError("platform worker did not complete its startup handshake")
            self.worker_startup = [item[1] for item in ready]
            self.actions = tuple(ready[0][2])
            self._refresh_worker_memory()
        except BaseException:
            self.close()
            raise

    def seed(self, seed: int | None = None) -> list[int | None]:
        self._seeds = [None if seed is None else int(seed) + index
                       for index in range(len(self.remotes))]
        return list(self._seeds)

    def reset(self) -> dict[str, np.ndarray]:
        for remote, seed, options in zip(self.remotes, self._seeds, self._options, strict=True):
            remote.send(("reset", (seed, options)))
        results = [remote.recv() for remote in self.remotes]
        observations, reset_infos = zip(*results, strict=True)
        self.reset_infos = list(reset_infos)
        self._seeds = [None] * len(self.remotes)
        self._options = [None] * len(self.remotes)
        return _stack_observations(observations)

    def step_async(self, actions: Sequence[int] | np.ndarray) -> None:
        for remote, action in zip(self.remotes, actions, strict=True):
            remote.send(("step", int(action)))
        self.waiting = True

    def step_wait(self) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray,
                                 tuple[dict[str, Any], ...]]:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        observations, rewards, dones, infos, reset_infos = zip(*results, strict=True)
        self.reset_infos = list(reset_infos)
        return (_stack_observations(observations), np.asarray(rewards, dtype=np.float32),
                np.asarray(dones, dtype=np.bool_), infos)

    def step(self, actions: Sequence[int] | np.ndarray) -> tuple[
            dict[str, np.ndarray], np.ndarray, np.ndarray, tuple[dict[str, Any], ...]]:
        self.step_async(actions)
        return self.step_wait()

    def get_attr(self, name: str) -> list[Any]:
        for remote in self.remotes:
            remote.send(("get_attr", name))
        return [remote.recv() for remote in self.remotes]

    def env_method(self, name: str, *args: Any, indices: int | Sequence[int] | None = None,
                   **kwargs: Any) -> list[Any]:
        payload = (name, args, kwargs)
        selected = self._selected_indices(indices)
        for index in selected:
            self.remotes[index].send(("env_method", payload))
        return [self.remotes[index].recv() for index in selected]

    def _selected_indices(self, indices: int | Sequence[int] | None) -> list[int]:
        if indices is None:
            return list(range(len(self.remotes)))
        if isinstance(indices, int):
            selected = [indices]
        else:
            selected = [int(index) for index in indices]
        if any(index < 0 or index >= len(self.remotes) for index in selected):
            raise IndexError("environment index out of range")
        return selected

    def _refresh_worker_memory(self) -> None:
        try:
            import psutil  # type: ignore[import-untyped]
        except ImportError:
            return
        for process, record in zip(self.processes, self.worker_startup, strict=True):
            if process.pid is None:
                continue
            try:
                worker = psutil.Process(process.pid)
                record["rss_bytes"] = int(worker.memory_info().rss)
                record["private_bytes"] = int(worker.memory_full_info().uss)
            except (psutil.Error, OSError):
                continue

    def set_attr(self, name: str, value: Any) -> None:
        for remote in self.remotes:
            remote.send(("set_attr", (name, value)))
        for remote in self.remotes:
            remote.recv()

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                try:
                    remote.recv()
                except (EOFError, OSError):
                    pass
            self.waiting = False
        for remote, process in zip(self.remotes, self.processes, strict=True):
            if process.is_alive():
                try:
                    remote.send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        for remote in self.remotes:
            remote.close()
        self.closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _stack_observations(observations: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.stack([observation[key] for observation in observations])
            for key in observations[0]}
