# 86h — pendências do 86g

Data da execução: 2026-11-01. Nenhum treino longo foi iniciado.

## Etapa 0 — percepção e entrada

- O preview usa limites screen-space e descarta quadros quase pretos. A captura manual de 86f foi repetida em duas sessões isoladas de `hk-ai-1`, com o dump completo de colliders habilitado na segunda. Ambas fecharam normalmente (`game_exit_code=0`, sem encerramento forçado) e restauraram o save original com SHA-256 confirmado.
- Sessão inicial `E:\VoidGymRuntime\86f-manual-capture\session-20261101-190911`: 910 solicitações em 270,2 s; cenas Menu_Title, Tutorial_01 e Town; os maiores rótulos HAZARD chegaram a 35,1% da grade, incorretamente cobrindo grande parte da imagem.
- Sessão diagnóstica `E:\VoidGymRuntime\86f-manual-capture\session-20261101-191627`: 1.150 solicitações em 332,4 s; 1.139 quadros não pretos (Menu_Title 414, Tutorial_01 570, Town 155). A máscara contém HAZARD em 130 quadros, EXIT em 52 e INTERACTIVE em nenhum. O maior componente HAZARD mede 370/1.296 células (28,55%), portanto a correção anterior não resolveu o falso bloco.
- O dump tem 620 colliders. Entre 21 classificados como HAZARD, 9 não têm `DamageHero` no próprio collider: oito sensores `Alert Range New` herdaram `hazardType`/nome de estalactites ancestrais e um sensor `Enemy Msg` de Cave Spikes. Os 12 hitboxes de perigo confirmados são `Attack` com `DamageHero` local (espinhos ou hitboxes poligonais de estalactite). A causa do excesso está demonstrada: `DamageHero` e o nome do perigo estavam sendo herdados de pais para colliders de detecção.
- O dump também confirma o colisor real do banco: objeto `RestBench`, com `RestBench` local, layer `Hero Detector`, trigger verdadeiro e área de 0,04% da tela. `Detect Range/RestBench` é outro trigger, herdando o componente do pai e ocupando 3,21%. A regra anterior rejeitava o trigger do banco e podia aceitar o detector do pai; isso explica o suporte INTERACTIVE zero mesmo com o banco presente.
- Correção aplicada: HAZARD exige `DamageHero` local e usa o `hazardType` desse mesmo componente; INTERACTIVE exige um componente de interação local reconhecido e aceita o trigger pequeno desse objeto. O limite de 8% continua rejeitando colliders grandes.
- Validação da correção: o executável de testes semânticos passou; a DLL Release compilou com 0 avisos/erros e foi implantada apenas em `hk-ai-1` pelo instalador com backup/hash. Um smoke isolado confirmou handshake, heartbeat, PING, captura 600×338, `RELEASE_ALL`, saída normal do jogo e transação de save concluída. SHA-256 implantado: `ec7799b522d3494b8ef6950d5589c448ef8ae3f524a8e915a3549cb928d07fc6`.
- Recaptura funcional com a DLL nova: `E:\VoidGymRuntime\86f-manual-capture\session-20261101-193731`; 727 quadros, 216,9 s, 716 não pretos (Menu_Title 21, Tutorial_01 542, Town 153). Finalização cooperativa, `game_exit_code=0`, sem término forçado e save restaurado por SHA-256.
- Resultado semântico nessa recaptura: HAZARD aparece em 94 quadros (4.097 células, 0,442% da amostra); maior componente 84/1.296 células (6,48%). INTERACTIVE aparece em 100 quadros (409 células, 0,044%); maior componente 6/1.296 (0,46%). A prévia mostra o rótulo pequeno sobre o banco em Dirtmouth e os espinhos em HAZARD. No dump completo, os 12 colliders marcados como perigo são `Attack` com `DamageHero` local; nenhum sensor `Alert Range New` ou `Enemy Msg` ficou HAZARD. Prévia tela/rótulo: `E:\VoidGymRuntime\86f-manual-capture\session-20261101-193731\label-validation-preview.png`.
- EXIT aparece em 45 quadros, mas ainda não foi inspecionado como alvo semântico; a validação desta rodada se limita a HAZARD e INTERACTIVE.
- O conjunto fixo de oito quadros do probe r2 foi usado tanto no gabarito quanto nas duas transferências zero-shot. Não contém HAZARD, ENEMY, EXIT ou INTERACTIVE, portanto não valida esses objetos; contém SOLID suficiente para medir a hipótese da rocha escura.
- Confusão SOLID nos mesmos oito quadros:
  - checkpoint `universal-009235520`: 53/5.357 células SOLID corretas (IoU 0,00956); 5.188 foram previstas como EMPTY.
  - checkpoint `universal-009255744`: 74/5.357 corretas (IoU 0,01323); 5.044 foram previstas como EMPTY.
  - Resultado: a hipótese da rocha escura se confirma; adicionamos o estilo de silhueta do ginásio.
- Novo estilo determinístico cobre exatamente 20% das sementes do teste de distribuição (200/1.000); no mosaico regenerado, 5/27 salas usam silhueta. Tem chão/plataformas quase pretos, borda clara, fundo médio iluminado e formas decorativas pretas em segundo plano, sem colisão. Mosaico e gabarito atualizados em `relatorios/86h_hk_probe_2026-11-01-r2/platform_gym_silhouette/`. Inspeção visual confirma que as máscaras semânticas continuam seguindo a geometria sólida, não os adornos.
- Entrada do modelo conferida: HK e PlatformGym entregam `(12,108,192)` uint8 RGB, valores 0–255, quatro imagens letterbox distintas após o buffer inicial; PNGs e manifests estão em `.../model_inputs/`. No HK, o passo 0 repete o frame inicial quatro vezes por falta de histórico; passos 1+ passam a conter quadros diferentes. `action_history` real altera a distribuição (TV médio 0,1906 nos cinco estados não iniciais), mas mudou a ação top-1 em 1/5 comparações. Tabela detalhada está em `relatorios/86h_hk_probe_2026-11-01-inputs/model_inputs/history_comparison.json`.
- Limite: as capturas de 86f anteriores foram geradas pela DLL antiga e não devem ser usadas como gabarito final de HAZARD/INTERACTIVE. A recaptura corrigida valida essas classes nesta rota. EXIT ainda requer revisão semântica. A presença do sprite do Knight sentado no banco foi capturada; não avaliamos alteração persistente do save, que foi restaurado após a sessão.

## Etapa 1 — política v2 em HK

Probe somente inferência, CUDA, checkpoint `universal-009235520`, 300 s após a inicialização: 1.044 decisões em `Tutorial_01`; 1 reset de episódio; duração total 304,4 s.

- 8 combinações de ataque ficaram liberadas em 1.044/1.044 decisões; nail arts ficaram indisponíveis, como esperado sem carga.
- Ataques escolhidos: 411/1.044 (39,37%); probabilidade softmax agregada média das ações de ataque: 37,10%.
- Com ATTACK enviado: 411; botões aplicados coincidiram com o pedido em 1.044/1.044 linhas. Tabela nome por nome do checkpoint e adaptador: idêntica (43 ações). Nenhum evento `hit_enemy` ocorreu; esta rodada comprova que a política escolheu e enviou ataques, mas não confirma contato/impacto em alvo. Portanto, máscara, tradução, tabela e ausência de ataques escolhidos não explicam a observação antiga; a causa de ausência de impacto continua inconclusiva.
- No ginásio, nível 2: 2.411/8.000 ações escolhidas eram ataques (30,14%). Esse arquivo contém frequência de ações escolhidas, não probabilidades softmax, então a comparação é de frequência, não de probabilidade.
- GIF de 30 s: `relatorios/86h_hk_probe_2026-11-01-r2/policy-30s.gif`. Registro por decisão e resumo em `...-r2/decisions.jsonl` e `summary.json`.
- Integridade: os hashes de `user1.dat`, backups e JSON da instância `hk-ai-1` coincidiram antes/depois. O jogo foi fechado ao fim; o save pessoal não foi usado.

## Etapa 2 — cabeça semântica e fumaça

Skip de alta resolução (~54×96), pesos recalculados, focal loss medida, compatibilidade com checkpoint antigo e testes já implementados. A fumaça CUDA r2 terminou em 20.224 passos efetivos a partir de `universal-009235520` (248,6 s): perda semântica 0,71131 → 0,00557; focal 0,42924 → 0,00115. O checkpoint `universal-009255744` foi salvo e recarregado com `semantic_skip` de forma `[16,54,96]`. A tentativa anterior, abortada pelo guard de memória, está registrada em `E:\VoidGymRuntime\v2\pretrain\86h-semantic-skip-smoke-2026-11-01`.

## Ainda pendente

1. Revisar/validar EXIT em quadros de transição antes de ampliar esse rótulo para outros cenários.
2. A rodada longa de currículo 0–8 segue aguardando a frase de autorização do operador. Não foi iniciada.
3. Etapa 1 deverá ser repetida após a rodada longa, conforme prompt.

## Validação

- Suíte completa: 871 passaram, 4 foram pulados, 3 avisos conhecidos e 8 subtests passaram (262,63 s). Testes focados atuais do renderer/estilos: 9 passaram.
- Ruff completo: passou. Mypy: 109 arquivos sem erros.
- `git diff --check`: sem erros; Git apenas alertou sobre normalização CRLF/LF.
- Teste C# semântico passou; build Release passou sem avisos; smoke isolado confirmou o carregamento da DLL em `hk-ai-1`; e a recaptura visual confirmou HAZARD/INTERACTIVE nos espinhos e no banco.
