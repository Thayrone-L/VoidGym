# 87v — beco sem saída na Passagem do Rei

**Estado em 06/11/2026:** o ginásio e suas avaliações CPU foram validados após a campanha `v2-hk-1` cair por commit baixo em 05/11. A campanha permanece parada; não aplicar o novo ginásio nela até uma nova execução autorizada.

## Campanha 87r

Na leitura de 05/11 às 10:36 BRT, o heartbeat do learner indicava `RUNNING`, rodada 47 e 12.462.592 passos concluídos. O `status.json` antigo ainda dizia `CANCELLED`, mas tinha timestamp de 04/11 às 21:42; não deve ser usado como estado atual. O learner registrou RAM disponível de cerca de 1,97 GiB e commit livre de cerca de 1,89 GiB. Por isso, enquanto a campanha segue, limitar o trabalho a código, checagens pequenas e leituras espaçadas; não usar GPU, rodar o ginásio ou a suíte completa.

Na leitura seguinte, em 05/11 às 10:58 BRT, `learner.json` e `heartbeat.json` estavam atualizados e indicavam `RUNNING`, rodada 65, 12.471.808 passos, RAM disponível ~2,75 GiB e commit livre ~1,69 GiB. O commit estava abaixo do guard de 3 GiB; mantive apenas trabalho de código e uma checagem de solver CPU curta, sem tocar no runtime.

Às 11:09 BRT, o heartbeat seguia `RUNNING`, rodada 72 e 12.475.392 passos. RAM disponível ~3,33 GiB; commit livre ~1,81 GiB, ainda abaixo do guard. Nenhuma ação foi enviada à campanha.

Às 11:13 BRT, o learner ainda estava `RUNNING`, rodada 76 e 12.477.440 passos. RAM disponível ~3,59 GiB; commit livre ~1,78 GiB. A campanha seguia progredindo e não foi alterada.

Às 11:48:02, o supervisor marcou `FAILED`: o commit corrente chegou a 1.444.663.296 bytes (cerca de 1,35 GiB), abaixo do piso de 1,5 GiB, enquanto o potencial estava em 18.624.532.480 bytes (cerca de 17,35 GiB). O learner e o supervisor somavam cerca de 1,89 GiB privados; o commit total usado era 32.733.958.144 bytes. A campanha não explica a maior parte do uso global de commit. O operador informou que Opera/serviços não estavam abertos no momento da falha e que só Codex e Claude estavam abertos; os dados históricos disponíveis não identificam qual processo consumia o restante do commit.

O fim da campanha e os testes posteriores foram separados: não houve comando meu nem worker persistente entre a leitura `RUNNING` das 11:13 e a falha das 11:48. A checagem de solver de 30 seeds, concluída em aproximadamente 0,50 s, ocorreu antes da leitura das 11:13. Depois da falha, as instâncias encerraram com `final_save_status=captured`; nenhum save pessoal foi lido ou alterado.

## O que significa progresso 131,37

Em `src/hk_agent/exploration_task.py`, cada cena guarda a primeira posição observada em `_scene_origins`. O valor de progresso é a maior distância euclidiana `hypot(x - origin_x, y - origin_y)` daquela origem, em unidades do jogo. Assim, 131,37 não é pontuação nem coordenada x absoluta. A repetição do mesmo máximo na v1 e nas duas instâncias da v2 é compatível com um limite geométrico no mesmo obstáculo; o número isolado não prova esse limite. Os registros da campanha situam a tentativa no trecho x≈164–167, y≈6–11, diante da parede.

## Isolamento da memória v1

O shim v2 em `src/voidgym/adapters/hollow_knight/runtime.py` substitui `lifelong_memory_root` pelo `shared_root` da campanha v2 e põe a saída de cada ator abaixo desse root. A configuração e o comando da campanha confirmam `shared_root=E:\VoidGymRuntime\shared\v2-hk-1` e `experience_root=E:\VoidGymRuntime\v2\experience`; as saídas de ator ficam em `shared\v2-hk-1\actors`. Isso isola, pelo caminho de execução, novidade vitalícia, conhecimento do jogador e pedidos de ajuda da instância v2. Nenhum arquivo de memória v1 nem save pessoal foi aberto nesta revisão; não encontrei referência a um caminho de memória v1 na configuração da campanha.

## Recompensas na parede

O código calcula `stuck_pressure` como penalidade limitada a 0,005 por transição quando o mesmo tile já acumulou ao menos três travadas. `stuck_escape_reward` só é elegível quando sai para um tile adjacente ainda não visitado, após três travadas no tile anterior; a concessão da ledger decai como `0,1 / sqrt(1 + visitas)`. A novidade do ExploreV4 contínuo é elegível apenas para tiles vitais novos, indexados em células de 4 unidades por cena. O custo temporal permanece ativo. As frequências e somas exatas dos últimos 3.000 passos na parede ainda precisam ser agregadas dos registros da campanha; não inferir efetividade a partir de uma linha isolada.

## Pedidos de ajuda

Os snapshots lidos anteriormente mostram pedidos `resolved_solo` em pontos anteriores do Tutorial_01 e um ponto com 25 travadas. A resolução automática atual ocorre ao mudar de cena ou quando o progresso da cena alcança pelo menos cinco unidades além da referência do pedido. Isso pode ser correto se o agente ultrapassou o ponto, mas não comprova que encontrou a rota. Correlacionar cada timestamp com posição e contagem por tile antes de classificar como falso positivo. Se o tile da parede chegar a dez travadas, manter um pedido aberto para o operador.

## Memória recorrente nas fronteiras `stuck`

Corrigido para as próximas campanhas: o ator mantém `next_state` e marca `episode_start=False` quando termina uma fatia PPO por `stuck`; apenas reset real do ambiente reinicializa a LSTM. O buffer já guarda esse estado por passo e o learner inicializa cada sequência a partir dele. O teste unitário da fronteira recorrente passou na suíte focada em CPU.

## Sala 9 — beco sem saída

Confirmada pelo operador a rota pretendida: recuar para a esquerda, subir as plataformas e atravessar por cima da parede. Há uma implementação em `generator.py`, `solver.py`, `adapter.py` e `curriculum.py`: três variantes determinísticas, parede sólida larga com altura 1,65× o pulo máximo, escada no lado de retorno, passarela acima da parede e saída do outro lado. A campanha v2-hk-1 terminou antes de usar essa alteração. A primeira checagem focada encontrou que o solver antigo segurava o pulo enquanto recuava e caía fora do mapa. O solver foi corrigido para dar pulos curtos mirando cada degrau, soltar o botão entre saltos e só então cruzar a passarela. A geração e a certificação por solver passaram em 30 seeds (cobrindo as três variantes) em cerca de 0,50 s; o caso espelhado direto também passou.

Na validação 87w, as três políticas simples dedicadas ao beco falharam em 200/200 seeds cada; o solver venceu 200/200. A política `universal-012438528` teve sucesso de 2% em 50 episódios determinísticos e 2% em 50 estocásticos (falha de 98% nos dois modos). Os resultados completos estão em `relatorios/87w_validacao/baselines_200.json` e `relatorios/87w_validacao/policy_012438528_level9_n50/behavior_progress.json` (resumo em `tabela.md`; `manifest_eval.json` registra a execução).

O nível 9 é liberado depois do nível 8, com sucesso-alvo de 70% para os níveis 0–9 no pré-treino 87v. A promoção do currículo agora aceita exatamente 70%, alinhada ao critério ≥70%; há teste focado para a fronteira. O nível 8 deve ser passado explicitamente como 70% no comando do pré-treino, para manter a regra “nenhum outro nível abaixo de 70%”. Nenhuma recompensa foi alterada.

## Pendências após a campanha

- Registrar estado final, passos até 131 unidades e tempo parado na parede para a comparação 87r.
- Correlacionar posições, recuos maiores que 5 u, ações e componentes de recompensa nos últimos 3.000 passos na parede; confirmar os pedidos de ajuda e o isolamento pelo manifesto da campanha.
- Registrar a interrupção da comparação 87r em 12.493.312 passos treinados (o `status.json` terminal estava atrasado em 12.439.040); os dados finais de jogabilidade e a comparação v1 seguem incompletos.
- A suíte completa passou (912 passed, 4 skipped, 3 warnings, 8 subtests); Ruff, `mypy src` e `git diff --check` passaram. Os commits por assunto foram enviados ao PR #8. Após corrigir dois timeouts externos dos testes Windows, os casos afetados passaram localmente e todos os checks do CI passaram no HEAD `f9fe927` (Code CI/Python, Bridge contracts, Portable policy boundary e dois jobs v2); detalhes no relatório 87w.
- Não iniciar o pré-treino 87v sem a autorização separada “autorizo o pré-treino 87v”. A campanha 87r também exige autorização explícita para retomar.
