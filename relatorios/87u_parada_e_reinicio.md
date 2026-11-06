# 87u — correções, parada segura e fumaça de 30 minutos

Data: 04/11/2026. A campanha longa `v2-hk-1` foi parada cooperativamente e suas evidências ficaram preservadas em `E:\VoidGymRuntime\shared\v2-hk-1-falha-interact-20261104`. O smoke isolado abaixo terminou pelo limite esperado de 30 minutos (`CANCELLED`), sem falha. Nenhuma campanha está ativa.

## Código e validação

As correções de 87s/87t e 87u foram separadas por assunto, commitadas e enviadas ao branch `codex/parallel-instances-isolation`, PR #8:

- `29ec631` Gate HK interact on visible targets
- `8c1156e` Keep HK actors live across stuck boundaries
- `49d51fc` Move learner contract to shared package
- Commits anteriores de 87s/87t no mesmo branch: `b27de4a`, `a86e823`, `ed41f59`, `6028028`, `16e25c2`.

O contrato foi movido para `src/voidgym_shared/learner_contract.py` e o import de v1 foi ajustado. Os cinco checks remotos do HEAD `49d51fc` estavam verdes. No HEAD `e557b35`, os workflows remotos `Code CI` e `VoidGym v2` também concluíram com sucesso.

A primeira execução da suíte identificou o gargalo no solver físico (cerca de 1,46 milhão de passos físicos para 1.800 salas). Otimizei `src/voidgym/gyms/platform_gym/solver.py` sem alterar certificados nem a asserção: os intervalos estáticos de piso são ordenados uma vez por sala, e direção/saída são calculadas uma vez por passo e reutilizadas pelo controlador. O teste de desempenho passou isoladamente em 70,75 s. Em seguida, a suíte completa passou: 902 aprovados, 4 ignorados, 3 avisos e 8 subtestes aprovados em 302,30 s. Ruff passou; `mypy src` passou sem erros em 113 arquivos; `git diff --check` passou. Os workflows `Code CI` e `VoidGym v2` do HEAD `e557b35` concluíram com sucesso.

## Auditoria das máscaras de ação

Pré-treino: 11.406 amostras dos blocos NPZ do checkpoint usado no 87u (`pretrain-86e-final-resume4-2026-11-03-cuda`), com máscaras descompactadas e contadas por nível 0–8. HK anterior à correção: 26.066 transições arquivadas em `v2-hk-1-falha-interact-20261104`. No JSONL do HK, `action_mask` enumera índices bloqueados; disponibilidade abaixo é o complemento desses índices. A amostra HK é do save usado e não representa habilidades não adquiridas.

| Ações (IDs da tabela) | Ginásio níveis 0–7 | Ginásio nível 8 | HK anterior, amostra |
|---|---:|---:|---:|
| 0–19: NOOP, LEFT, RIGHT, JUMP_TAP, JUMP_HOLD, LEFT+JUMP_TAP, LEFT+JUMP_HOLD, RIGHT+JUMP_TAP, RIGHT+JUMP_HOLD, JUMP_MAX, LEFT+JUMP_MAX, RIGHT+JUMP_MAX, ATTACK, LEFT+ATTACK, RIGHT+ATTACK, UP+ATTACK, DOWN+ATTACK, JUMP+ATTACK, LEFT+JUMP+ATTACK, RIGHT+JUMP+ATTACK | 100% cada | 100% cada | 100% cada |
| 20–34: CAST, UP+CAST, DOWN+CAST, FOCUS_HOLD, NAIL_ART_CHARGE, LEFT+NAIL_ART_CHARGE, RIGHT+NAIL_ART_CHARGE, UP+NAIL_ART_CHARGE, DOWN+NAIL_ART_CHARGE, DASH, LEFT+DASH, RIGHT+DASH, DOWN+DASH, SUPER_DASH_HOLD, DREAM_NAIL | 0% cada | 0% cada | 0% cada na amostra |
| 35: INTERACT | 0% | 0,45% | 100% |
| 36–42: MENU_UP, MENU_DOWN, MENU_LEFT, MENU_RIGHT, MENU_SUBMIT, MENU_CANCEL, OPEN_CHARMS | 0% cada | 0% cada | 0% cada na amostra |

Assim, `INTERACT` esteve disponível em 0/11.406 amostras nos níveis 0–7 e em 24 amostras do nível 8; agregado: 0,21%. Na HK antiga, esteve liberada em 26.066/26.066 passos e foi selecionada em 10.766/26.066 (41,30%). Essa foi a discrepância material. As ações básicas estavam 100% liberadas nos dois conjuntos; as ações de habilidade/menu ficaram 0% na amostra HK, que não tinha habilidades ou menus disponíveis.

A correção v2 em `src/voidgym/adapters/hollow_knight/adapter.py:50-64,136-145` libera `INTERACT` só quando existe prompt validado ou rótulo `INTERACTIVE` até duas células Chebyshev do jogador. A v1 não foi alterada. O teste `test_tutorial_01_interact_is_masked_away_and_allowed_near_bench` cobre longe/perto do alvo e rejeita prompt não validado. Limitação: `task-transitions.jsonl` é escrito pelo task interno v1 e não registra a máscara externa do adaptador v2; portanto o smoke não mede diretamente essa máscara em jogo real. O teste valida a regra do adaptador, e a ação `INTERACT` não foi escolhida durante o smoke.

## Causa do reset e correção de continuidade

Antes do commit `8c1156e`, `src/voidgym/training/actor.py` chamava `env.reset()` para qualquer `terminated` ou `truncated`. `stuck` chega como truncamento. O reset de `AdapterEnv` delega a `HollowKnightAdapter.reset`, que delega ao task v1; `ExplorationTask.reset` chama `self.environment.reset()`. Esse caminho aciona o reset do ambiente HK. Dependendo da configuração, o backend reutiliza a bridge em jogada contínua ou usa bootstrap/reabertura. Nos 26.066 passos arquivados da campanha antiga, houve 71 fins por `stuck` (33 no slot 0 e 38 no slot 1, cerca de 35 por ator), sem morte real. Os episódios sucessivos terminam em `Tutorial_01` e as posições recomeçam perto do spawn (aprox. x=35), coerentes com a volta ao início que o operador viu. O manifesto antigo não registra `reset_recovery.method`; por isso não dá para afirmar se a volta ao início foi feita por menu ou por reabertura do save. O acionamento indevido do reset de ambiente após stuck está identificado.

Desde `8c1156e`, `stuck` encerra apenas a fatia PPO. O ator mantém observação, posição e instância do jogo, sem `env.reset()` ou bootstrap de menu/save; reinicializa apenas o estado recorrente do modelo na fronteira PPO. O truncamento ainda recebe bootstrap de valor PPO para calcular retornos. `test_two_stuck_slices_keep_same_live_environment_without_reset` verifica duas fronteiras consecutivas sem chamadas de reset.

## Smoke 87u — 30 minutos, dois atores

Campanha isolada `v2-hk-1-smoke-87u-30m-20261104`: início 20:11:44 BRT, fim 20:41:49 BRT (30 min 05 s), `CANCELLED` pelo prazo, conforme esperado. Ao fim, painel com `active_count=0`, slots cancelados e saves em raízes isoladas por instância. Checkpoint inicial `universal-012438528`; learner terminou em `universal-012444160`: 5.632 passos efetivos, 11 rodadas. Os atores registraram 2.827 e 2.832 passos; o JSONL contém 5.659 transições.

O prompt 87u informa cerca de 300 ms por decisão antes da correção; a execução antiga não registrava a métrica de inferência atual, então essa comparação não é instrumentada. Agora, slot 0: média 33,29 ms, p95 57,08 ms; slot 1: média 30,65 ms, p95 47,34 ms. Ambos usaram duas threads. Meta de inferência <60 ms atingida. Vazão fim a fim: 1,57 passo/s por ator (2.827/1.800 s e 2.832/1.800 s), abaixo da meta aproximada de 7 passos/s. Espera, ponte e ambiente continuam limitando a vazão.

### Frequência de seleção por ação

Contagens sobre as 5.659 linhas de `task-transitions.jsonl`; todas as 43 ações estão listadas. São seleções observadas, não disponibilidade de máscara.

| Ação | Contagem | Frequência |
|---|---:|---:|
| NOOP | 118 | 2,09% |
| LEFT | 72 | 1,27% |
| RIGHT | 61 | 1,08% |
| JUMP_TAP | 334 | 5,90% |
| JUMP_HOLD | 89 | 1,57% |
| LEFT+JUMP_TAP | 346 | 6,11% |
| LEFT+JUMP_HOLD | 8 | 0,14% |
| RIGHT+JUMP_TAP | 54 | 0,95% |
| RIGHT+JUMP_HOLD | 11 | 0,19% |
| JUMP_MAX | 35 | 0,62% |
| LEFT+JUMP_MAX | 292 | 5,16% |
| RIGHT+JUMP_MAX | 1.362 | 24,07% |
| ATTACK | 150 | 2,65% |
| LEFT+ATTACK | 304 | 5,37% |
| RIGHT+ATTACK | 965 | 17,05% |
| UP+ATTACK | 106 | 1,87% |
| DOWN+ATTACK | 761 | 13,45% |
| JUMP+ATTACK | 312 | 5,51% |
| LEFT+JUMP+ATTACK | 96 | 1,70% |
| RIGHT+JUMP+ATTACK | 183 | 3,23% |
| CAST | 0 | 0,00% |
| UP+CAST | 0 | 0,00% |
| DOWN+CAST | 0 | 0,00% |
| FOCUS_HOLD | 0 | 0,00% |
| NAIL_ART_CHARGE | 0 | 0,00% |
| LEFT+NAIL_ART_CHARGE | 0 | 0,00% |
| RIGHT+NAIL_ART_CHARGE | 0 | 0,00% |
| UP+NAIL_ART_CHARGE | 0 | 0,00% |
| DOWN+NAIL_ART_CHARGE | 0 | 0,00% |
| DASH | 0 | 0,00% |
| LEFT+DASH | 0 | 0,00% |
| RIGHT+DASH | 0 | 0,00% |
| DOWN+DASH | 0 | 0,00% |
| SUPER_DASH_HOLD | 0 | 0,00% |
| DREAM_NAIL | 0 | 0,00% |
| INTERACT | 0 | 0,00% |
| MENU_UP | 0 | 0,00% |
| MENU_DOWN | 0 | 0,00% |
| MENU_LEFT | 0 | 0,00% |
| MENU_RIGHT | 0 | 0,00% |
| MENU_SUBMIT | 0 | 0,00% |
| MENU_CANCEL | 0 | 0,00% |
| OPEN_CHARMS | 0 | 0,00% |

### Fronteiras stuck e avanço

Dos 5.659 passos, 89 fecharam uma fatia com `truncated=true` e `termination_reason=stuck` (62 slot 0, 27 slot 1); ocorreram também 6 mortes reais (`terminated=true`, `player_death`) no slot 0. O smoke não criou um novo episódio de jogo para cada stuck: o diário mostrou seis vidas encerradas por morte no slot 0 e uma vida aberta no slot 1. Isso é consistente com a nova regra de manter o ambiente nas fronteiras stuck; o teste automatizado verifica diretamente duas fronteiras sem reset.

Máximos de x por vida registrada: slot 0, episódios 1–6: 121,84; 118,05; 148,49; 146,20; 140,89; 154,39 (spawn aprox. x=35,46). Slot 1, episódio 1: 10,61 (spawn x=4,27). Slot 0 avançou até x=154,39; o slot 1 também se moveu no jogo, como mostrado ao operador, mas os referenciais não são diretamente comparáveis. O smoke não salvou x por fronteira PPO, então esses são máximos por vida, não por fatia.

Na amostra final do heartbeat: RAM disponível 3,61 GiB, commit disponível 2,48 GiB, commit potencial disponível 18,49 GiB. São valores finais, não mínimos durante os 30 minutos.

Métricas finais do learner: `L_hero_motion=0.7654`, `L_inverse_dynamics=1.4210`, `L_ppo=-0.0210`, `L_semantic=0.3662`, `L_semantic_focal=0.1597`, entropia 1.5608, explained variance -1.2400, KL 0.0170, loss 0.3664.

## Próximo passo e evidências

Os números do smoke não são comparáveis à v1. O 87u concluiu a parada segura, as correções, a auditoria e o smoke de 30 minutos. A campanha longa do 87r, do zero a partir de `universal-012438528`, não foi iniciada. O 87u pede autorização explícita para reiniciá-la; a vazão também está abaixo da meta. Aguardar decisão do operador.

- Smoke: `E:\VoidGymRuntime\shared\v2-hk-1-smoke-87u-30m-20261104\status.json`, `learner.json`, `heartbeat.json`, `telemetry\actor-actor-slot-0.json`, `telemetry\actor-actor-slot-1.json`.
- Transições e progresso: `E:\VoidGymRuntime\shared\v2-hk-1-smoke-87u-30m-20261104\actors\actor-slot-*\training\task-transitions.jsonl`; `journey\campaign-progress.json`.
- Pré-treino: `E:\VoidGymRuntime\v2\pretrain\pretrain-86e-final-resume4-2026-11-03-cuda\experience\platform_gym\pretrain-86e-final-resume4-2026-11-03-cuda\blocks`.
- HK anterior: `E:\VoidGymRuntime\shared\v2-hk-1-falha-interact-20261104\actors\actor-slot-*\training\task-transitions.jsonl`.
- Código: `src/voidgym/adapters/hollow_knight/adapter.py:50-64,136-145`; `src/voidgym/training/actor.py:69,167-180,227,326-328`; `src/voidgym/envs/adapter_env.py:64-80`; `src/hk_agent/exploration_task.py:626-647`; `src/hk_agent/supervisor_backend.py:228-330`.
- Testes: `tests/v2/test_hk_adapter.py::test_tutorial_01_interact_is_masked_away_and_allowed_near_bench`; `tests/v2/training/test_recurrent_training.py::test_actor_keeps_live_environment_across_stuck_slices`; `tests/test_exploration_task.py::test_two_stuck_slices_keep_same_live_environment_without_reset`.
