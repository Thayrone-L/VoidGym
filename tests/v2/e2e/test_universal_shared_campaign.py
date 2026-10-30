from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from tests.v2.fake_adapter import FakeAdapter
from voidgym.envs.adapter_env import AdapterEnv
from voidgym.models.policy import UniversalPolicy
from voidgym.training.actor import run_actor
from voidgym.training.checkpoints import load_universal, save_universal
from voidgym.training.shared_learner import run_shared_learner


def test_universal_learner_updates_checkpoint_and_two_actors_import_it(tmp_path: Path) -> None:
    actions = FakeAdapter().action_table()
    checkpoint = tmp_path / "pretrained"
    save_universal(checkpoint, UniversalPolicy(action_count=len(actions)), actions, steps=0)
    shared = tmp_path / "shared"
    stop_file = tmp_path / "STOP"
    actors = ("actor-a", "actor-b")

    def actor(actor_id: str) -> dict[str, object]:
        env = AdapterEnv(FakeAdapter(), lambda _before, _after: 0.0)
        policy = UniversalPolicy(action_count=len(actions))
        return run_actor(
            env, policy, shared_root=shared, actor_id=actor_id,
            action_table=actions, universal_checkpoint=checkpoint,
            experience_root=tmp_path / "experience" / actor_id,
            campaign_id="e2e-universal", rollout_steps=2, sequence_length=1,
            wait_timeout=30, stop_file=stop_file,
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        actor_jobs = [pool.submit(actor, actor_id) for actor_id in actors]
        learner_job = pool.submit(
            run_shared_learner,
            shared_root=shared, universal_checkpoint=checkpoint,
            action_table=actions, actors=actors, total_timesteps=4,
            rollout_steps=2, sequence_length=1, learning_rate=1e-4,
            lr_encoder_scale=0.3, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, entropy_coef=0.01, epochs=1, batch_size=1,
            c_sem=0.0, c_inv=0.0, c_mot=0.0, device_name="cpu",
            wait_timeout=30,
        )
        learner_result = learner_job.result(timeout=60)
        stop_file.touch()
        actor_results = [job.result(timeout=60) for job in actor_jobs]

    assert learner_result["status"] == "COMPLETED"
    assert learner_result["completed_timesteps"] == 4
    assert json.loads((shared / "learner.json").read_text(encoding="utf-8"))["status"] == "COMPLETED"
    assert all(result["completed_steps"] >= 2 for result in actor_results)
    latest = shared / "latest.json"
    assert latest.is_file()
    latest_data = json.loads(latest.read_text(encoding="utf-8"))
    assert latest_data["round"] == 1
    policy_dir = shared / latest_data["policy"]
    imported = UniversalPolicy(action_count=len(actions))
    loaded = load_universal(policy_dir, imported, actions)
    assert loaded["trained_steps"] == 4
