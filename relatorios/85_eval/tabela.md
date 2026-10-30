# Avaliação do pré-treino v2

| Modo | Nível | Episódios | Sucesso | Passos médios | Mortes | Travadas |
|---|---:|---:|---:|---:|---:|---:|
| deterministic | 0 | 100 | 76.0% | 275.8 | 0 | 15 |
| deterministic | 1 | 100 | 69.0% | 354.3 | 0 | 17 |
| deterministic | 2 | 100 | 62.0% | 616.9 | 0 | 0 |
| deterministic | 3 | 100 | 10.0% | 448.3 | 0 | 79 |
| deterministic | 4 | 100 | 1.0% | 347.0 | 0 | 96 |
| deterministic | 5 | 100 | 74.0% | 393.9 | 14 | 6 |
| deterministic | 6 | 100 | 18.0% | 601.4 | 101 | 61 |
| deterministic | 7 | 30 | 0.0% | 662.7 | 0 | 22 |
| stochastic | 0 | 100 | 81.0% | 277.8 | 0 | 13 |
| stochastic | 1 | 100 | 78.0% | 353.8 | 0 | 8 |
| stochastic | 2 | 100 | 62.0% | 614.6 | 0 | 0 |
| stochastic | 3 | 100 | 46.0% | 448.6 | 0 | 41 |
| stochastic | 4 | 100 | 45.0% | 566.7 | 0 | 37 |
| stochastic | 5 | 100 | 69.0% | 512.6 | 55 | 4 |
| stochastic | 6 | 100 | 72.0% | 544.2 | 74 | 1 |
| stochastic | 7 | 30 | 3.3% | 1231.9 | 4 | 10 |

## IoU e matriz de confusão (5.000 quadros balanceados, split test)

| Classe | IoU |
|---|---:|
| EMPTY | 0.893 |
| SOLID | 0.377 |
| PLATFORM | 0.168 |
| HAZARD | 0.000 |
| ENEMY | 0.000 |
| PLAYER | 0.297 |
| BREAKABLE | 0.000 |
| PICKUP | 0.000 |
| INTERACTIVE | n/d |
| EXIT | 0.066 |

Amostra: 625 quadros determinísticos por nível (0–7). Matriz completa: `confusion.csv`.
Prévia: `percepcao.png`. GIFs: `gifs/nivel_3.gif`, `gifs/nivel_4.gif`, `gifs/nivel_6.gif`.
