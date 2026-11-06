# Avaliação do pré-treino v2

| Modo | Nível | Episódios | Sucesso | Passos médios | Mortes | Travadas |
|---|---:|---:|---:|---:|---:|---:|
| deterministic | 9 | 50 | 2.0% | 373.6 | 0 | 46 |
| stochastic | 9 | 50 | 2.0% | 521.8 | 0 | 48 |

## Ações escolhidas por nível

Contagens completas por ação e episódio estão em `behavior_progress.json`.

| Modo | Nível | Ações mais frequentes |
|---|---:|---|
| deterministic | 9 | RIGHT+ATTACK: 5782, LEFT+JUMP_MAX: 4460, RIGHT+JUMP_MAX: 3652, LEFT+ATTACK: 2715, ATTACK: 945, NOOP: 570, DOWN+ATTACK: 295, JUMP_TAP: 181 |
| stochastic | 9 | RIGHT+ATTACK: 6063, LEFT+JUMP_MAX: 5843, RIGHT+JUMP_MAX: 3596, DOWN+ATTACK: 3547, JUMP_TAP: 1198, LEFT+ATTACK: 861, JUMP+ATTACK: 783, ATTACK: 714 |

Percepção não executada nesta chamada; consulte a avaliação balanceada separada.
