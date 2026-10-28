"""Run an isolated session whose gameplay captures also persist semantic labels.

The bootstrap patches only this process's imported control server at runtime, so
the v1 source tree remains unchanged while the bridge's optional label contract
is requested for every saved frame.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BOOTSTRAP = r'''
import os
import sys
from pathlib import Path

if os.environ.get("VOIDGYM_86C_CAPTURE_SEMANTICS") == "1":
    project_src = os.environ["VOIDGYM_86C_PROJECT_SRC"]
    if project_src not in sys.path:
        sys.path.insert(0, project_src)
    import hk_agent.gameplay_control as gameplay_control
    from hk_agent.supervisor import require_png_frame

    async def capture_with_semantics(self, label):
        frame = await self._supervisor.capture_once(
            timeout=self._command_timeout_seconds, bridge=self._bridge,
            episode_id=self._episode_id, semantic_labels=True,
        )
        self._capture_index += 1
        path = self._output_dir / f"{self._capture_index:04d}-{label}.png"
        require_png_frame(frame)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(frame.data)
        temporary.replace(path)
        semantic_path = path.with_suffix(".sem.u4")
        if frame.semantic_labels_u4 is None:
            raise RuntimeError("bridge did not return the requested semantic labels")
        semantic_path.write_bytes(frame.semantic_labels_u4)
        telemetry = frame.operator_telemetry
        visible_enemies = (
            telemetry.get("visible_enemies", [])[:6]
            if isinstance(telemetry, dict) and isinstance(telemetry.get("visible_enemies", []), list)
            else []
        )
        record = {
            "index": self._capture_index, "path": str(path),
            "semantic_labels_path": str(semantic_path), "semantic_codec": "u4-v1",
            "frame_id": frame.frame_id, "capture_monotonic_ns": frame.capture_monotonic_ns,
            "width": frame.width, "height": frame.height, "bytes": len(frame.data),
            "scene_transition": frame.scene_transition, "duplicate_of": frame.duplicate_of,
            "last_applied_action_id": frame.last_applied_action_id,
            "scene_name": frame.scene_name, "visible_enemies": visible_enemies,
            "bestiary_events": list(frame.bestiary_events),
            "hp_lost_since_last_frame": frame.hp_lost_since_last_frame,
        }
        self.captures.append(record)
        return record

    gameplay_control.GameplayControlServer._capture = capture_with_semantics
'''


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run run_isolated_session.py with semantic labels saved beside each gameplay frame"
    )
    parser.add_argument("session_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    forwarded = list(args.session_args)
    if forwarded and forwarded[0] == "--":
        forwarded.pop(0)
    if not forwarded:
        parser.error("pass the run_isolated_session.py arguments after --")

    with tempfile.TemporaryDirectory(prefix="voidgym-86c-semantic-") as directory:
        Path(directory, "sitecustomize.py").write_text(BOOTSTRAP, encoding="utf-8")
        environment = os.environ.copy()
        existing_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (directory, str(PROJECT_ROOT / "src"), existing_path) if part
        )
        environment["VOIDGYM_86C_CAPTURE_SEMANTICS"] = "1"
        environment["VOIDGYM_86C_PROJECT_SRC"] = str(PROJECT_ROOT / "src")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "tools" / "run_isolated_session.py"), *forwarded],
            check=False,
            env=environment,
        )
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
