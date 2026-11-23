# Avaliação do pré-treino v2

| Modo | Nível | Episódios | Sucesso | Passos médios | Mortes | Travadas |
|---|---:|---:|---:|---:|---:|---:|
| deterministic | 0 | 100 | 93.0% | 93.0 | 0 | 2 |
| deterministic | 1 | 100 | 85.0% | 73.3 | 0 | 14 |
| deterministic | 2 | 100 | 54.0% | 457.4 | 0 | 21 |
| deterministic | 3 | 100 | 86.0% | 77.2 | 0 | 12 |
| deterministic | 4 | 100 | 76.0% | 100.2 | 0 | 23 |
| deterministic | 5 | 100 | 95.0% | 39.5 | 2 | 4 |
| deterministic | 6 | 100 | 5.0% | 432.9 | 1122 | 93 |
| deterministic | 7 | 30 | 33.3% | 780.2 | 1 | 6 |
| stochastic | 0 | 100 | 93.0% | 118.4 | 0 | 0 |
| stochastic | 1 | 100 | 99.0% | 28.8 | 0 | 0 |
| stochastic | 2 | 100 | 59.0% | 627.6 | 0 | 0 |
| stochastic | 3 | 100 | 99.0% | 32.2 | 0 | 0 |
| stochastic | 4 | 100 | 90.0% | 169.8 | 0 | 1 |
| stochastic | 5 | 100 | 98.0% | 43.7 | 5 | 0 |
| stochastic | 6 | 100 | 94.0% | 172.3 | 164 | 0 |
| stochastic | 7 | 30 | 60.0% | 634.8 | 0 | 0 |

## IoU e matriz de confusão

| Classe | IoU | Suporte |
|---|---:|---:|
| EMPTY | 0.953 | 6125568 |
| SOLID | 0.182 | 244347 |
| PLATFORM | 0.090 | 18725 |
| HAZARD | 0.000 | 0 |
| ENEMY | 0.000 | 0 |
| PLAYER | 0.684 | 30042 |
| BREAKABLE | 0.000 | 0 |
| PICKUP | nan | 0 |
| INTERACTIVE | nan | 0 |
| EXIT | 0.090 | 61318 |

Matriz completa: `confusion.csv`.
Prévia: `percepcao.png`.
GIFs: `gifs/nivel_3.gif`, `gifs/nivel_4.gif`, `gifs/nivel_6.gif`.
