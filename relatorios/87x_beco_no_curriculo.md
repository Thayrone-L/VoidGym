# 87x — beco no currículo e pré-treino

## Escopo e estado

- Branch: `codex/parallel-instances-isolation`, PR #8.
- Checkpoint de referência: `universal-012438528` em `E:\VoidGymRuntime\v2\pretrain\pretrain-86e-final-resume4-2026-11-03-cuda`.
- Nenhuma campanha HK nova foi iniciada. O prompt exige autorização textual específica para `v2-hk-2`; ela ainda não foi dada.
- O pré-treino 87v foi autorizado pelo operador anteriormente (“autorizo o pré-treino 87v quando chegar o momento”); o início fica condicionado à suíte e CI verde.

## Etapa 1 — currículo

- Alvo padrão do nível 8 restaurado para 30%; níveis 0–7 continuam em 70% e o beco (nível 9) em 70%.
- A partir do currículo nível 3, o nível 9 recebe 15% dos episódios. Essa cota é retirada dos 60% do nível atual: ficam 45% no nível atual, 30% nos níveis inferiores e 10% no próximo nível.
- Quando a janela móvel de 100 episódios do nível 9 alcança 70%, a cota cai para 5% (o nível atual passa a 55%). Abaixo da meta ou sem 100 episódios, volta a 15%.
- A promoção existente continua usando a janela e o limiar de 70%; a passagem 0→8 não foi alterada.

## Etapa 2 — salas de beco e separação treino/teste

- Largura varia entre 24 e 40 unidades; posição da parede, largura (2–4 u), altura (1,4–2,0× pulo máximo), 2–5 degraus, subida (0,35–0,6×), espaçamento (1,8–3,2 u), distância inicial (1,5–8 u) e altura da ponte variam dentro das restrições físicas.
- Variantes i–iii e espelhamento foram mantidos. A variante iv usa sala de 40 u e uma primeira escada 17–19 u atrás do spawn, fora do semicírculo visível inicial; usa dois degraus para caber e continuar alcançável.
- O split `train` exclui a combinação de parede de 4 u e 5 degraus. `test` reserva essa combinação em 25% das salas das variantes i–iii e também contém salas não extremas.
- A geração só retorna salas certificadas pelo solver. Nas 200 salas do split teste, o solver venceu 200/200. As políticas bobas tiveram taxa de falha de 99,5% cada: `FORWARD`, `FORWARD+JUMP_MAX` e `FORWARD+ATTACK`.
- Artefatos: `relatorios/87x_beco_no_curriculo/dumb-test.json`.
- Resultado da amostra de teste: `FORWARD`, `FORWARD+JUMP_MAX` e `FORWARD+ATTACK` passaram 1/200 cada; taxa de falha de 99,5% por política.

## Referência — `universal-012438528` no beco

50 episódios por modo e split, limite de 1.500 passos por episódio. A avaliação foi somente do comportamento do nível 9; percepção e os demais níveis serão avaliados após o pré-treino.

| Split | Modo | Sucesso | `stuck` | Média de passos |
|---|---|---:|---:|---:|
| teste | determinístico | 4% (2/50) | 47 | 318,32 |
| teste | estocástico | 12% (6/50) | 41 | 507,20 |
| treino | determinístico | 2% (1/50) | 46 | 381,24 |
| treino | estocástico | 10% (5/50) | 43 | 531,90 |

O checkpoint de referência está abaixo da meta do beco em ambos os splits, como esperado pela revisão. Os artefatos de comportamento estão em `relatorios/87x_beco_no_curriculo/baseline-012438528-{train,test}/`.

## Etapa 3 — pré-treino 87v

Pendente: iniciar de `universal-012438528` com supervisor, retomada, guard de memória, checkpoints e avaliações rápidas; manter `memory-history.jsonl` com os 10 maiores processos a cada cinco minutos. Não iniciar até a suíte completa, Ruff, mypy, `git diff --check` e CI remoto ficarem verdes.

## Etapa 4 — campanha HK nova

Pendente de autorização textual específica: **“autorizo iniciar a campanha v2-hk-2”**. Não iniciar a campanha sem essa frase.

## Validação e commits

- Suíte local: `919 passed, 4 skipped, 8 subtests passed`; os 3 avisos são de `stable-baselines3` ao construir tensores de outros tensores.
- Ruff (`ruff check .`), `mypy src` e `git diff --check`: aprovados.
- CI remoto, commits por etapa e push ao PR #8: pendentes.
- Resultados do pré-treino e comparação tripla serão acrescentados aqui quando concluídos. A comparação tripla depende de autorização da Etapa 4.
