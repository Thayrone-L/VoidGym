from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def create_installer_logger(root: Path) -> tuple[logging.Logger, Path]:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    path = root / f"install-{stamp}.log"
    logger = logging.getLogger(f"voidgym.installer.{path.stem}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger, path
