# Avaliação do pré-treino v2

| Modo | Nível | Episódios | Sucesso | Passos médios | Mortes | Travadas |
|---|---:|---:|---:|---:|---:|---:|
| deterministic | 9 | 50 | 2.0% | 381.2 | 0 | 46 |
| stochastic | 9 | 50 | 10.0% | 531.9 | 0 | 43 |

## Ações escolhidas por nível

Contagens completas por ação e episódio estão em `behavior_progress.json`.

| Modo | Nível | Ações mais frequentes |
|---|---:|---|
| deterministic | 9 | LEFT+JUMP_MAX: 5681, RIGHT+ATTACK: 4433, RIGHT+JUMP_MAX: 4156, ATTACK: 2671, LEFT+ATTACK: 1204, NOOP: 380, DOWN+ATTACK: 298, JUMP_TAP: 227 |
| stochastic | 9 | LEFT+JUMP_MAX: 5697, RIGHT+ATTACK: 5327, RIGHT+JUMP_MAX: 4629, DOWN+ATTACK: 2987, JUMP_TAP: 1442, ATTACK: 1247, LEFT+ATTACK: 1157, LEFT+JUMP_TAP: 877 |

Percepção não executada nesta chamada; consulte a avaliação balanceada separada.
