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

Pronto para iniciar após a CI do commit atual ficar verde:

- Origem: `universal-012438528`, com supervisor e até três retomadas automáticas em 24 h.
- Limites: 16 h ou 3 milhões de passos sem melhora desde o início deste pré-treino; avaliações rápidas a cada 1 milhão de passos, 50 episódios por nível.
- O marco inicial do observador de estagnação fica salvo em `progress-watch-origin.json` por runtime. Assim, os marcadores antigos carregados do checkpoint-base não encerram uma campanha nova imediatamente, enquanto retomadas do mesmo runtime preservam o mesmo limite acumulado.
- O guard do pré-treino segue a proteção 87w: aguarda quando o commit atual fica abaixo de 1,5 GiB, publicando `waiting_memory` no heartbeat; retoma quando recuperar. Falha após 30 minutos contínuos abaixo do mínimo ou imediatamente se o commit potencial cair abaixo de 3 GiB.
- O supervisor grava `memory-history.jsonl` no runtime: top 10 processos por memória privada (nome, PID, memória privada e working set) e métricas de commit, inicialmente e a cada cinco minutos.
- O ponto de avaliação final será níveis 0–9, split de teste do beco e percepção, comparando com `universal-012438528`; nenhum outro nível pode cair mais de cinco pontos.
- Primeira tentativa em 11/11/2026: o guard inicial foi implementado por engano com mínimo atual de 3 GiB e encerrou após 30 minutos, embora o commit potencial permanecesse acima de 15 GiB. O learner fez uma atualização PPO de 2.048 passos (12.438.528 → 12.440.576) antes do guard; nenhum checkpoint foi salvo além do checkpoint-base.
- Segunda tentativa, runtime `pretrain-87v-beco-2026-11-11-cuda-r1`: o guard corrigido aguardou a recuperação do commit atual e o learner fez uma atualização (2.048 passos), salvando `universal-012440576`. Encerrou em 11/11/2026 às 14:35:56 por `level 2 had no rolling-success improvement for 3000000 steps`: o checkpoint-base carregava `last_improvement_step=9437376`, mais de 3 milhões abaixo do passo inicial 12.438.528. Esse limite histórico antecedia o 87v; não correspondeu a 3 milhões de passos sem melhora durante esta nova campanha. A retomada r1 foi preservada como evidência e não será usada como origem.
- Correção antes da nova tentativa: cada runtime 87v registra seu passo-base em `progress-watch-origin.json` e inicializa o limite de estagnação a partir desse marco. Retomadas do mesmo runtime mantêm esse marco e o estado de progresso; o checkpoint-base permanece exatamente `universal-012438528`.
- Na saída da r1: status `COMPLETED` por limite herdado; duração 6m08s, total 12.440.576, nível de currículo 8, nível 2 em 42% na janela móvel de 100, sem checkpoint íntegro além do `universal-012440576` salvo pela tentativa curta. RAM disponível 2,65 GiB, commit atual disponível 1,57 GiB e potencial 14,53 GiB; não houve erro CUDA. O nível 9 ainda não tinha episódios na janela.
- Terceira tentativa, runtime `pretrain-87v-beco-2026-11-11-cuda-r2`: nenhum learner iniciou. O plano conservador de recursos falhou antes de alcançar o guard de espera: com commit atual abaixo da reserva necessária para calcular ambientes, `plan_resources` encerrou três tentativas automáticas. A memória potencial estava segura, mas a espera de 30 minutos ainda só ocorria depois do planejamento.
- Ajuste seguinte aplicado: a espera CUDA acontece antes do planejamento e se repete depois da sondagem do worker, sempre recalculando o hardware após a espera. O piso de planejamento é 10 GiB, cobrindo 4 GiB do sistema, 5,5 GiB do learner e uma pequena margem do worker; o limite durante o treinamento continua em 1,5 GiB atual / 3 GiB potencial. A espera continua limitada a 30 minutos. A validação focada passou (29 testes).

## Etapa 4 — campanha HK nova

Pendente de autorização textual específica: **“autorizo iniciar a campanha v2-hk-2”**. Não iniciar a campanha sem essa frase.

## Validação e commits

- Suíte local após os ajustes do watchdog e da espera pré-planejamento: `925 passed, 4 skipped, 8 subtests passed`; os 3 avisos são de `stable-baselines3` ao construir tensores de outros tensores.
- Ruff nos arquivos alterados, `mypy src` e `git diff --check`: aprovados após o ajuste.
- CI do commit `f501bf4` passou em todos os jobs após um retry de e2e que inicialmente expirou em 60 s. O teste e2e passou no retry em 7m01s.
- Commits enviados ao PR #8: `6ef3ac7` (currículo), `e9326e6` (salas e avaliações), `84bf7fa` (telemetria), `2c69203` (tolerância do teste de desempenho), `c9b7d02` (primeira versão do guard), `15e23f7` (limiar do guard) e `f501bf4` (watchdog de estagnação por runtime). O ajuste de espera pré-planejamento será acrescentado.
- Resultados do pré-treino e comparação tripla serão acrescentados aqui quando concluídos. A comparação tripla depende de autorização da Etapa 4.
