# 87w — falha por commit baixo, correções e retomada pendente

**Atualizado:** 06/11/2026 (BRT). **Campanha:** `v2-hk-1`, parada em `FAILED`. Não retomei nem reiniciei o treino.

## 1. Falha e consumo

Às 11:13 de 05/11 o learner ainda estava `RUNNING`, rodada 76, em 12.477.440 passos, com RAM livre de ~3,59 GiB e commit livre de ~1,78 GiB. Às 11:48:02 o supervisor marcou `FAILED`: commit corrente 1.444.663.296 bytes (~1,35 GiB), abaixo do piso de 1,5 GiB. O potencial era 18.624.532.480 bytes (~17,35 GiB), acima do piso de 3 GiB. RAM livre no último batimento: 2.241.454.080 bytes (~2,09 GiB).

O snapshot do learner registrava 2.030.964.736 bytes (~1,89 GiB) de memória privada somada aos processos da campanha e 32.733.958.144 bytes de commit usado no sistema. Jogos, learner, atores e supervisor não explicam a maior parte do commit do sistema. O `status.json` terminal ficou desatualizado em 12.439.040 passos/rodada 1; o checkpoint completo mais novo era `universal-000012493312` (12.493.312 passos, rodada 107). Esse checkpoint foi verificado pelo manifesto e arquivos de pesos, e é o ponto de retomada previsto. Os registros dos atores mostram `final_save_status=captured`; nenhum save pessoal foi lido ou alterado.

O operador informou que, na hora da falha, somente Codex e Claude estavam abertos; Opera e serviços não estavam abertos. Não há telemetria histórica de memória privada por processo que identifique a origem do restante do commit. Não atribuo a falha a uma aplicação sem essa evidência.

Entre a leitura das 11:13 e a falha das 11:48, não executei comandos nem deixei workers ativos. A rodada de solver de 30 seeds (~0,50 s) e as pequenas checagens Python ocorreram antes da leitura das 11:13; não houve avaliação, suíte ou pré-treino nesse intervalo. Elas não são uma causa provável para a pressão que persistiu no host.

Snapshot atual do host em 06/11 às 11:40 BRT: RAM livre 3,91 GiB; commit livre 12,40 GiB. Não havia Hollow Knight, ator, learner nem supervisor da campanha ativos. Os dez maiores processos por memória privada nessa leitura:

| Processo | PID | Privada (GiB) | Working set (GiB) |
|---|---:|---:|---:|
| vmmem | 24456 | 4,01 | 1,26 |
| NVIDIA Broadcast | 16068 | 0,82 | 0,24 |
| opera | 24304 | 0,60 | 0,17 |
| ChatGPT | 22300 | 0,52 | 0,30 |
| ChatGPT | 23176 | 0,48 | 0,45 |
| opera | 18040 | 0,43 | 0,56 |
| claude | 6612 | 0,41 | 0,48 |
| MsMpEng | 5280 | 0,41 | 0,33 |
| Creative Cloud UI Helper | 3876 | 0,34 | 0,33 |
| ChatGPT | 24412 | 0,31 | 0,25 |

Essa é uma leitura atual; não descreve os processos em 05/11 às 11:48.

## 2. Correções implementadas

- O painel agora inicia o supervisor v2 com `--max-restarts 3` (até três reinícios em 24 horas, quatro tentativas no total).
- O learner espera antes de iniciar e entre rodadas quando o commit corrente fica abaixo de 1,5 GiB. Durante a espera mantém `RUNNING`, publica `waiting_memory=true` em `learner.json`/`heartbeat.json` e volta ao treino quando a margem recupera. Falha imediatamente se o commit potencial conhecido cair abaixo de 3 GiB, ou após 30 minutos contínuos abaixo do piso corrente. Um pedido de parada encerra a espera.
- No fim `FAILED` ou `CANCELLED`, o supervisor combina learner/latest/checkpoints completos e escreve os passos, rodada, política e métricas mais recentes no `status.json`; a leitura é somente leitura e valida que a política fique dentro do runtime.
- A faixa de estado do painel usa “Precisa de atenção” e a falha permanece como banner de erro mesmo quando há alertas de ajuda/travada. Não existe mecanismo de notificação do sistema operacional no painel; o mecanismo disponível é o banner e o histórico de eventos.

## 3. Avaliações 87v concluídas

- Solver e geração do beco: 200/200 salas do nível 9 solucionadas.
- Baselines: `FORWARD`, `FORWARD+JUMP_MAX` e `FORWARD+ATTACK` falharam em 200/200 cada (100% de falha).
- Checkpoint `universal-012438528`: nível 9 em 50 episódios por modo. Determinístico: 2% sucesso, média 373,64 passos, 46/50 episódios marcados `stuck`; estocástico: 2% sucesso, média 521,76 passos, 48/50 `stuck`. Ambos superam o limiar de 80% de falha.
- O script de baselines tinha um erro ao compilar resultados do nível 9, pois usava nomes de políticas dos níveis anteriores. Corrigi o cálculo para usar as políticas da sala corrente e repeti a avaliação completa. Resultado: `relatorios/87w_validacao/baselines_200.json`. Resultado da política: `relatorios/87w_validacao/policy_012438528_level9_n50/behavior_progress.json` e `tabela.md`.

## 4. Testes, validações e commits

Os testes focados e a suíte completa passaram: **912 passed, 4 skipped, 3 warnings, 8 subtests**, em 276,97 s na execução final. `ruff check .`, `mypy src` (114 arquivos) e `git diff --check` passaram. O primeiro CI após o push passou nos checks Bridge contracts, Portable policy boundary e nos dois jobs v2; o job Python falhou apenas no teste `test_wait_rollouts_rejects_rollout_from_retired_actor_launch`, cujo timeout de 10 ms expirou antes da gravação do log no runner Windows. Aumentei esse timeout para 1 s; o teste isolado e a suíte completa passaram localmente. O CI será repetido com essa correção antes de considerar a revisão concluída. Os commits locais separados são: 87w `f5c324b`; memória recorrente `26ad0ad`; solver `86519ec`; geometria `6769333`; currículo `c2f2074`; ferramentas e relatório 87v `9b0f085`; relatório 87u `7235b85`. A integração do commit remoto de 87u e as correções locais foi enviada ao PR #8; a correção do teste seguirá em novo commit.

## 5. Retomada

O runtime continua parado em `FAILED`. Não houve retomada. O prompt exige a frase explícita **“autorizo retomar a campanha 87”**; ela ainda não foi fornecida. A campanha deve retomar de `universal-000012493312`, sem zerar os passos, com a mesma configuração e o save contínuo das instâncias. A mudança do ator recorrente valerá para os passos depois da retomada e precisa ser registrada. O pré-treino 87v continua dependendo da autorização separada **“autorizo o pré-treino 87v”**.
