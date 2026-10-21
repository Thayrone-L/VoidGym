"""Perfis falsos e cálculos previsíveis de ResourcePlan."""

import pytest

from voidgym.core.config import ResourceOverrides
from voidgym.core.hardware import HardwareProfile, detect_hardware, plan_resources


def _profile(physical: int, logical: int, ram: float, gpu: float | None) -> HardwareProfile:
    return HardwareProfile(
        physical,
        logical,
        ram,
        ram / 2,
        "Fake NVIDIA" if gpu else None,
        gpu,
        gpu - 1 if gpu else None,
        "8.6" if gpu else None,
        200.0,
        "Windows",
    )


@pytest.mark.parametrize(
    ("profile", "device", "instances", "envs", "batch"),
    [
        (_profile(8, 16, 16, 12), "cuda", 3, 8, 2048),
        (_profile(4, 8, 8, None), "cpu", 1, 4, 512),
        (_profile(16, 32, 64, 24), "cuda", 3, 24, 2048),
    ],
)
def test_plans_for_reference_machines(profile, device, instances, envs, batch) -> None:
    plan = plan_resources(profile, "game_training")
    assert (plan.train_device, plan.game_instances, plan.gym_envs, plan.batch_size) == (
        device,
        instances,
        envs,
        batch,
    )
    assert plan.actor_device == "cpu"
    assert plan.torch_threads_actor == 1


def test_real_machine_detection_and_overrides() -> None:
    profile = detect_hardware()
    assert profile.logical_cores and profile.disk_free_gb is not None
    plan = plan_resources(
        profile, "pretrain_platform", ResourceOverrides(gym_envs=3, train_device="cpu")
    )
    assert plan.gym_envs == 3
    assert plan.train_device == "cpu"
    with pytest.raises(ValueError, match="train_device"):
        plan_resources(profile, "evaluation", ResourceOverrides(train_device="tpu"))
