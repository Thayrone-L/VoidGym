"""Buffer de transições com gabaritos privilegiados e máscara de descontinuidade."""

from dataclasses import dataclass, field
from typing import cast

import numpy as np


@dataclass
class TrainingBuffer:
    images: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    semantic: list[np.ndarray] = field(default_factory=list)
    hero_motion: list[np.ndarray] = field(default_factory=list)
    discontinuity: list[bool] = field(default_factory=list)
    privileged: list[np.ndarray] = field(default_factory=list)
    labeled: list[bool] = field(default_factory=list)

    def add(self, image: np.ndarray, action: int, info: dict[str, object]) -> None:
        labels = info.get("training_labels")
        valid = isinstance(labels, dict) and "sem" in labels
        label_dict = cast(dict[str, object], labels) if isinstance(labels, dict) else {}
        self.images.append(np.asarray(image, dtype=np.uint8).copy())
        self.actions.append(int(action))
        self.semantic.append(np.asarray(label_dict.get("sem", np.full((27, 48), 255)), dtype=np.uint8))
        if isinstance(labels, dict):
            velocity = np.asarray(label_dict.get("hero_velocity", (np.nan, np.nan)), dtype=np.float32)
            self.hero_motion.append(np.asarray((*velocity, label_dict.get("on_ground", np.nan)), dtype=np.float32))
        else:
            self.hero_motion.append(np.full(3, np.nan, np.float32))
        self.discontinuity.append(bool(info.get("discontinuity", False)))
        self.privileged.append(np.asarray(info.get("privileged", (np.nan,) * 4), dtype=np.float32))
        self.labeled.append(valid)

    def clear(self) -> None:
        self.__dict__.update(TrainingBuffer().__dict__)
