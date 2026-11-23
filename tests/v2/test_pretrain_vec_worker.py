from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_spawned_platform_worker_does_not_import_torch(tmp_path: Path) -> None:
    project = Path(__file__).parents[2]
    snippet = f"""
import json, sys
from functools import partial
sys.path.insert(0, {str(project / 'src')!r})
from voidgym.gyms.platform_gym.vec_worker import make_platform_env
from voidgym.gyms.platform_gym.vector_env import PlatformSubprocVecEnv
if __name__ == '__main__':
    env = PlatformSubprocVecEnv([partial(make_platform_env, level=0, seed=3)])
    print(json.dumps(env.worker_startup[0]))
    env.close()
"""
    result = subprocess.run([sys.executable, "-c", snippet], cwd=project, check=True,
                            capture_output=True, text=True, timeout=60)
    record = json.loads(result.stdout.strip())
    assert record["torch_imported"] is False
    assert record["private_bytes"] > 0
    assert record["rss_bytes"] > 0
