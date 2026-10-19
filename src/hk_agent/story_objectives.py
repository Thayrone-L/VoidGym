from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StoryObjective:
    """A small, progression ordered objective revealed by an in-game progress flag."""

    objective_id: str
    reveal_flag: str
    kind: str
    target: str
    target_flag: str | None
    scene_prefixes: tuple[str, ...]
    completion_reward: float = 2.0
    arrival_reward: float = 0.5


STORY_OBJECTIVES: tuple[StoryObjective, ...] = (
    StoryObjective(
        "reach_crossroads", "enteredTutorialFirstTime", "reach_area", "Crossroads",
        None, ("Crossroads",),
    ),
    StoryObjective(
        "defeat_false_knight", "metElderbug", "defeat_boss", "False Knight",
        "falseKnightDefeated", (),
    ),
    StoryObjective(
        "reach_greenpath", "falseKnightDefeated", "reach_area", "Greenpath",
        None, ("Fungus1", "Greenpath"),
    ),
)

OBJECTIVE_BY_ID = {item.objective_id: item for item in STORY_OBJECTIVES}


def is_target_scene(objective: StoryObjective, scene: str) -> bool:
    folded = scene.casefold()
    return any(folded.startswith(prefix.casefold()) for prefix in objective.scene_prefixes)
