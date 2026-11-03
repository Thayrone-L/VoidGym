# Avaliação do pré-treino v2

| Modo | Nível | Episódios | Sucesso | Passos médios | Mortes | Travadas |
|---|---:|---:|---:|---:|---:|---:|
| deterministic | 0 | 200 | 99.0% | 23.9 | 0 | 0 |
| deterministic | 1 | 200 | 67.0% | 108.6 | 0 | 66 |
| deterministic | 2 | 200 | 50.5% | 638.2 | 0 | 20 |
| deterministic | 3 | 200 | 90.5% | 44.5 | 0 | 18 |
| deterministic | 4 | 200 | 46.5% | 171.7 | 0 | 107 |
| deterministic | 5 | 200 | 76.5% | 212.0 | 428 | 27 |
| deterministic | 6 | 200 | 97.0% | 30.4 | 46 | 6 |
| deterministic | 7 | 200 | 79.5% | 263.0 | 108 | 12 |
| deterministic | 8 | 200 | 35.0% | 362.6 | 99 | 112 |
| stochastic | 0 | 200 | 96.0% | 69.8 | 0 | 0 |
| stochastic | 1 | 200 | 100.0% | 13.1 | 0 | 0 |
| stochastic | 2 | 200 | 60.5% | 607.8 | 0 | 0 |
| stochastic | 3 | 200 | 99.0% | 25.8 | 0 | 0 |
| stochastic | 4 | 200 | 95.5% | 83.7 | 0 | 0 |
| stochastic | 5 | 200 | 94.0% | 111.2 | 19 | 0 |
| stochastic | 6 | 200 | 99.5% | 32.2 | 63 | 0 |
| stochastic | 7 | 200 | 93.5% | 130.3 | 39 | 0 |
| stochastic | 8 | 200 | 71.0% | 513.4 | 63 | 0 |

## Ações escolhidas por nível

Contagens completas por ação e episódio estão em `behavior_progress.json`.

| Modo | Nível | Ações mais frequentes |
|---|---:|---|
| deterministic | 0 | RIGHT+JUMP_MAX: 1263, LEFT+ATTACK: 1178, ATTACK: 1016, NOOP: 446, LEFT+JUMP_MAX: 433, JUMP_TAP: 249, RIGHT+ATTACK: 97, LEFT+JUMP_TAP: 86 |
| deterministic | 1 | LEFT+JUMP_MAX: 14287, RIGHT+JUMP_MAX: 7252, RIGHT+ATTACK: 174 |
| deterministic | 2 | LEFT+ATTACK: 58038, ATTACK: 41202, NOOP: 12788, JUMP_TAP: 4532, RIGHT+JUMP_MAX: 3725, DOWN+ATTACK: 2421, JUMP+ATTACK: 1298, RIGHT+ATTACK: 1282 |
| deterministic | 3 | LEFT+JUMP_MAX: 5081, RIGHT+JUMP_MAX: 2155, LEFT+ATTACK: 777, ATTACK: 355, NOOP: 352, RIGHT+ATTACK: 167, DOWN+ATTACK: 8, JUMP_TAP: 2 |
| deterministic | 4 | RIGHT+JUMP_MAX: 33872, RIGHT+ATTACK: 438, LEFT+JUMP_MAX: 28, DOWN+ATTACK: 3, RIGHT+JUMP_TAP: 1 |
| deterministic | 5 | LEFT+ATTACK: 14113, ATTACK: 9445, LEFT+JUMP_MAX: 9173, NOOP: 4748, RIGHT+JUMP_MAX: 2058, JUMP_TAP: 1423, RIGHT+ATTACK: 666, DOWN+ATTACK: 565 |
| deterministic | 6 | RIGHT+ATTACK: 3477, RIGHT+JUMP_MAX: 1507, LEFT+JUMP_MAX: 879, JUMP_TAP: 66, ATTACK: 60, LEFT+ATTACK: 48, DOWN+ATTACK: 31, LEFT+JUMP_TAP: 10 |
| deterministic | 7 | LEFT+ATTACK: 21102, ATTACK: 14414, NOOP: 5413, RIGHT+ATTACK: 3778, LEFT+JUMP_MAX: 3421, JUMP_TAP: 2076, RIGHT+JUMP_MAX: 1602, DOWN+ATTACK: 519 |
| deterministic | 8 | RIGHT+JUMP_MAX: 18588, LEFT+ATTACK: 17883, LEFT+JUMP_MAX: 13613, INTERACT: 7038, ATTACK: 5181, RIGHT+ATTACK: 4557, DOWN+ATTACK: 2495, NOOP: 1982 |
| stochastic | 0 | ATTACK: 4215, LEFT+ATTACK: 2329, JUMP_TAP: 1742, LEFT+JUMP_TAP: 1523, RIGHT+JUMP_MAX: 1227, NOOP: 603, LEFT+JUMP_MAX: 408, LEFT: 329 |
| stochastic | 1 | RIGHT+JUMP_MAX: 1443, LEFT+JUMP_MAX: 628, RIGHT+ATTACK: 328, DOWN+ATTACK: 50, JUMP_TAP: 37, RIGHT+JUMP+ATTACK: 25, JUMP+ATTACK: 18, LEFT+ATTACK: 18 |
| stochastic | 2 | ATTACK: 39839, LEFT+ATTACK: 24407, JUMP_TAP: 17461, LEFT+JUMP_TAP: 14852, NOOP: 6055, DOWN+ATTACK: 3335, LEFT: 3212, RIGHT+JUMP+ATTACK: 2295 |
| stochastic | 3 | RIGHT+JUMP_MAX: 1297, ATTACK: 1118, LEFT+ATTACK: 583, JUMP_TAP: 448, LEFT+JUMP_MAX: 423, RIGHT+ATTACK: 294, LEFT+JUMP_TAP: 292, NOOP: 188 |
| stochastic | 4 | ATTACK: 4537, LEFT+ATTACK: 2731, JUMP_TAP: 2077, RIGHT+JUMP_MAX: 2051, LEFT+JUMP_TAP: 1772, RIGHT+ATTACK: 756, NOOP: 732, DOWN+ATTACK: 481 |
| stochastic | 5 | ATTACK: 6201, LEFT+ATTACK: 3792, JUMP_TAP: 2781, LEFT+JUMP_TAP: 2318, RIGHT+JUMP_MAX: 1319, NOOP: 999, RIGHT+ATTACK: 989, DOWN+ATTACK: 890 |
| stochastic | 6 | RIGHT+ATTACK: 1852, RIGHT+JUMP_MAX: 1690, LEFT+JUMP_MAX: 665, ATTACK: 522, LEFT+ATTACK: 329, JUMP_TAP: 292, DOWN+ATTACK: 248, LEFT+JUMP_TAP: 218 |
| stochastic | 7 | ATTACK: 6687, LEFT+ATTACK: 4207, JUMP_TAP: 3113, LEFT+JUMP_TAP: 2422, RIGHT+JUMP_MAX: 1654, RIGHT+ATTACK: 1612, DOWN+ATTACK: 1207, NOOP: 1148 |
| stochastic | 8 | ATTACK: 26504, LEFT+ATTACK: 18226, JUMP_TAP: 13090, LEFT+JUMP_TAP: 11444, RIGHT+JUMP_MAX: 6057, RIGHT+ATTACK: 4976, DOWN+ATTACK: 4762, NOOP: 4732 |

Percepção não executada nesta chamada; consulte a avaliação balanceada separada.
