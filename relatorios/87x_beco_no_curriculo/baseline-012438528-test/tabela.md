# Avaliação do pré-treino v2

| Modo | Nível | Episódios | Sucesso | Passos médios | Mortes | Travadas |
|---|---:|---:|---:|---:|---:|---:|
| deterministic | 9 | 50 | 4.0% | 318.3 | 0 | 47 |
| stochastic | 9 | 50 | 12.0% | 507.2 | 0 | 41 |

## Ações escolhidas por nível

Contagens completas por ação e episódio estão em `behavior_progress.json`.

| Modo | Nível | Ações mais frequentes |
|---|---:|---|
| deterministic | 9 | LEFT+JUMP_MAX: 6455, RIGHT+ATTACK: 5165, RIGHT+JUMP_MAX: 2684, ATTACK: 1474, DOWN+ATTACK: 111, LEFT+ATTACK: 16, JUMP+ATTACK: 4, JUMP_TAP: 3 |
| stochastic | 9 | LEFT+JUMP_MAX: 6171, RIGHT+ATTACK: 4981, RIGHT+JUMP_MAX: 3115, DOWN+ATTACK: 2882, JUMP_TAP: 1513, ATTACK: 1438, LEFT+ATTACK: 1340, LEFT+JUMP_TAP: 800 |

Percepção não executada nesta chamada; consulte a avaliação balanceada separada.
