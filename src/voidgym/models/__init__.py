"""Modelos universais compartilhados entre jogos (prompt 85)."""

from voidgym.models.aux_heads import HeroMotionHead, InverseDynamicsHead
from voidgym.models.encoder import ImpalaEncoder
from voidgym.models.perception import SemanticHead
from voidgym.models.policy import UniversalPolicy

__all__ = ["HeroMotionHead", "ImpalaEncoder", "InverseDynamicsHead", "SemanticHead", "UniversalPolicy"]
