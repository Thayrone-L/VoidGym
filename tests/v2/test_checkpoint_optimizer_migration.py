from __future__ import annotations

from pathlib import Path

import torch

from voidgym.models.policy import UniversalPolicy
from voidgym.training.checkpoints import load_universal, save_universal


def test_single_saved_group_migrates_to_current_encoder_groups(tmp_path: Path) -> None:
    action_table = tuple(f"action-{index}" for index in range(43))
    source = UniversalPolicy()
    old_optimizer = torch.optim.Adam(source.parameters(), lr=2.5e-4)
    torch.stack([parameter.square().sum() for parameter in source.parameters()]).sum().backward()  # type: ignore[no-untyped-call]
    old_optimizer.step()
    checkpoint = tmp_path / "old-checkpoint"
    save_universal(checkpoint, source, action_table, steps=123, optimizer=old_optimizer)

    restored = UniversalPolicy()
    encoder_ids = {id(parameter) for parameter in restored.encoder.parameters()}
    encoder = [parameter for parameter in restored.parameters() if id(parameter) in encoder_ids]
    other = [parameter for parameter in restored.parameters() if id(parameter) not in encoder_ids]
    optimizer = torch.optim.Adam([{"params": encoder}, {"params": other}], lr=1e-3)

    manifest = load_universal(checkpoint, restored, action_table, optimizer=optimizer)

    state = optimizer.state_dict()
    assert manifest["trained_steps"] == 123
    assert len(state["param_groups"]) == 2
    assert [len(group["params"]) for group in state["param_groups"]] == [len(encoder), len(other)]
    assert len(state["state"]) == len(list(restored.parameters()))
