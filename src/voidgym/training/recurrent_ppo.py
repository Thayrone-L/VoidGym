"""Primitivas PPO recorrenti e perdas auxiliares em minibatches."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional

from voidgym.models.perception import semantic_loss


@dataclass
class LossWeights:
    semantic: float = 0.5
    inverse: float = 0.1
    motion: float = 0.1


def ppo_objective(new_log_prob: torch.Tensor, old_log_prob: torch.Tensor, advantage: torch.Tensor,
                  value: torch.Tensor, returns: torch.Tensor, entropy: torch.Tensor,
                  clip: float = 0.2, ent_coef: float = 0.01) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratio = (new_log_prob - old_log_prob).exp()
    policy = -torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage).mean()
    value_loss = functional.mse_loss(value, returns)
    total = policy + 0.5 * value_loss - ent_coef * entropy.mean()
    with torch.no_grad():
        kl = ((ratio - 1) - (new_log_prob - old_log_prob)).mean()
    return total, {"L_ppo": total.detach(), "entropy": entropy.mean().detach(), "KL": kl}


def auxiliary_objective(logits: torch.Tensor, labels: torch.Tensor, inverse_logits: torch.Tensor,
                        actions: torch.Tensor, motion_pred: torch.Tensor,
                        motion_target: torch.Tensor, discontinuity: torch.Tensor,
                        weights: LossWeights | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or LossWeights()
    valid_sem = labels.ne(255)
    sem = semantic_loss(logits, labels) if valid_sem.any() else logits.sum() * 0
    valid_inv = ~discontinuity.bool()
    inv = functional.cross_entropy(inverse_logits[valid_inv], actions[valid_inv].long()) if valid_inv.any() else inverse_logits.sum() * 0
    valid_motion = torch.isfinite(motion_target).all(-1)
    mot = functional.mse_loss(motion_pred[valid_motion], motion_target[valid_motion]) if valid_motion.any() else motion_pred.sum() * 0
    total = weights.semantic * sem + weights.inverse * inv + weights.motion * mot
    return total, {"L_semantic": sem.detach(), "L_inverse_dynamics": inv.detach(), "L_hero_motion": mot.detach()}


def augment_sequence(images: torch.Tensor, semantic: torch.Tensor, dx: int, dy: int,
                     brightness: float = 0.0, contrast: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate a complete batch consistently with replicated image borders and shifted labels."""
    moved = torch.roll(images, shifts=(dy, dx), dims=(-2, -1))
    if dy > 0:
        moved[..., :dy, :] = images[..., :1, :].expand(*images.shape[:-2], dy, images.shape[-1])
    elif dy < 0:
        moved[..., dy:, :] = images[..., -1:, :].expand(*images.shape[:-2], -dy, images.shape[-1])
    if dx > 0:
        moved[..., :, :dx] = images[..., :, :1].expand(*images.shape[:-2], images.shape[-2], dx)
    elif dx < 0:
        moved[..., :, dx:] = images[..., :, -1:].expand(*images.shape[:-2], images.shape[-2], -dx)
    moved = ((moved.float() * contrast + brightness * 255).clamp(0, 255)).to(images.dtype)
    sem_dx, sem_dy = round(dx / 4), round(dy / 4)
    sem = torch.roll(semantic, shifts=(sem_dy, sem_dx), dims=(-2, -1))
    if sem_dy > 0:
        sem[..., :sem_dy, :] = 255
    elif sem_dy < 0:
        sem[..., sem_dy:, :] = 255
    if sem_dx > 0:
        sem[..., :, :sem_dx] = 255
    elif sem_dx < 0:
        sem[..., :, sem_dx:] = 255
    return moved, sem


class RecurrentPPO:
    """Otimizador compartilhado; coleta de ambientes é controlada pelo executor."""

    def __init__(self, model: nn.Module, lr: float = 2.5e-4, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.device = torch.device(device)

    def update(self, loss: torch.Tensor) -> float:
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        return float(loss.detach().cpu())
