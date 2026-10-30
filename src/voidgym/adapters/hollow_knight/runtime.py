"""Construct the production Hollow Knight v1 reward shim for a v2 actor."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

# V1-SHIM: portar no prompt 86. Reuse the managed explore_v4 task outside the v1 control plane.
from hk_agent.campaign_source import (
    CampaignBootstrapSequence,
    CampaignOrigin,
    resolve_campaign_bootstrap,
)
from hk_agent.config import load_config
from hk_agent.exploration_task import ExplorationTaskConfig
from tools.run_first_training import build_task


def create_hollow_knight_env(args: Any, *, shared_root: Path, actor_id: str,
                             slot_id: int,
                             request_semantic_labels: bool = True) -> tuple[Any, Any]:
    """Create AdapterEnv + owner using the managed slot and explore_v4 task shim."""
    config_path = Path(args.config).resolve(strict=True)
    project_config = load_config(config_path, allow_training=True)
    slot = slot_id
    if slot < 0 or slot >= len(project_config.install_roots):
        raise ValueError("v2 Hollow Knight actor requires a reserved managed slot")
    if (Path(args.game_install).resolve(strict=True) != project_config.install_roots[slot]
            or Path(args.instance_data_root).resolve(strict=False) != project_config.instance_data_roots[slot]):
        raise ValueError("v2 actor paths do not match the reserved managed slot")
    if args.reward_version != "explore_v4":
        raise ValueError("the v2 Hollow Knight adapter currently requires explore_v4")
    origin = CampaignOrigin(args.save_source, args.save_slot)
    initial = resolve_campaign_bootstrap(
        continue_plan_path=Path(args.bootstrap_plan),
        new_game_plan_path=Path(args.new_game_bootstrap_plan),
        origin=origin,
    )
    subsequent = (resolve_campaign_bootstrap(
        continue_plan_path=Path(args.bootstrap_plan),
        new_game_plan_path=Path(args.new_game_bootstrap_plan),
        origin=CampaignOrigin("existing_save", args.save_slot),
    ) if origin.source == "new_game" else None)
    bootstrap_sequence = CampaignBootstrapSequence(
        initial.plan, subsequent.plan if subsequent is not None else None
    )
    task_config = ExplorationTaskConfig(
        reward_version="explore_v4", continuous_playthrough=args.save_source == "continuous_playthrough",
        horizon_steps=args.horizon, novelty_reward=args.novelty_reward,
        novelty_threshold=args.novelty_threshold, novelty_budget=args.novelty_budget,
        scene_change_reward=args.scene_change_reward, enemy_kill_reward=args.enemy_kill_reward,
        enemy_kill_reward_budget=args.enemy_kill_reward_budget, shade_recovery_reward=4.0,
        damage_penalty=args.damage_penalty, geo_gain_reward=0.0, geo_spend_reward=0.0,
        stuck_seconds=args.stuck_seconds, evaluation_interval=args.evaluation_interval,
        tutorial_hint_level=args.tutorial_hint_level, death_penalty=args.death_penalty,
        run_failed_penalty=args.run_failed_penalty, time_cost=args.time_cost,
        lease_base_ms=args.lease_ms, ent_coef=args.ent_coef, gamma=args.gamma,
        gae_lambda=args.gae_lambda, n_epochs=args.n_epochs, target_kl=args.target_kl,
        learning_rate=args.learning_rate, clip_range=args.clip_range,
    )
    task_config.validate()
    # Preserve the complete CLI namespace expected by the v1 task factory;
    # v2 only overrides the memory root and fixes the task to ExploreV4.
    task_args = Namespace(**vars(args))
    task_args.task = "exploration"
    task_args.lifelong_memory_root = shared_root
    task_root = shared_root / "actors" / actor_id
    task, owner = build_task(
        project_config=project_config, bootstrap_plan=initial.plan,
        bootstrap_plan_provider=bootstrap_sequence, output_root=task_root,
        task_config=task_config, args=task_args, phase="training",
        request_semantic_labels=request_semantic_labels,
        request_combat_events=True,
        request_breakable_labels=True,
    )
    from voidgym.adapters.hollow_knight.adapter import HollowKnightAdapter
    from voidgym.envs.adapter_env import AdapterEnv

    adapter = HollowKnightAdapter(task, enable_discovery_reward=True)
    env = AdapterEnv(adapter, lambda _before, _after: adapter.last_reward)
    action_names = getattr(task, "_action_names", ())
    if tuple(env.actions) != tuple(action_names):
        owner.close()
        raise ValueError("v2 adapter action table differs from explore_v4")
    return env, owner
