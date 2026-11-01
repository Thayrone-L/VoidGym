# 86h — pendências do 86g

Data da execução: 2026-11-01. Nenhum treino longo foi iniciado.

## Etapa 0 — percepção e entrada

- O preview corrigido usa limites screen-space; gatilhos grandes (>8% da tela) não são tratados como perigo sem indicação visual específica, interativos usam colliders sólidos pequenos, regiões grandes geram aviso e quadros quase pretos são descartados. A captura nova de política não teve quadros descartados na cena observada.
- O conjunto fixo de oito quadros do probe r2 foi usado tanto no gabarito quanto nas duas transferências zero-shot. Não contém HAZARD, ENEMY, EXIT ou INTERACTIVE, portanto não valida esses objetos; contém SOLID suficiente para medir a hipótese da rocha escura.
- Confusão SOLID nos mesmos oito quadros:
  - checkpoint `universal-009235520`: 53/5.357 células SOLID corretas (IoU 0,00956); 5.188 foram previstas como EMPTY.
  - checkpoint `universal-009255744`: 74/5.357 corretas (IoU 0,01323); 5.044 foram previstas como EMPTY.
  - Resultado: a hipótese da rocha escura se confirma; adicionamos o estilo de silhueta do ginásio.
- Novo estilo determinístico cobre exatamente 20% das sementes do teste de distribuição (200/1.000); no mosaico regenerado, 5/27 salas usam silhueta. Tem chão/plataformas quase pretos, borda clara, fundo médio iluminado e formas decorativas pretas em segundo plano, sem colisão. Mosaico e gabarito atualizados em `relatorios/86h_hk_probe_2026-11-01-r2/platform_gym_silhouette/`. Inspeção visual confirma que as máscaras semânticas continuam seguindo a geometria sólida, não os adornos.
- Entrada do modelo conferida: HK e PlatformGym entregam `(12,108,192)` uint8 RGB, valores 0–255, quatro imagens letterbox distintas após o buffer inicial; PNGs e manifests estão em `.../model_inputs/`. No HK, o passo 0 repete o frame inicial quatro vezes por falta de histórico; passos 1+ passam a conter quadros diferentes. `action_history` real altera a distribuição (TV médio 0,1906 nos cinco estados não iniciais), mas mudou a ação top-1 em 1/5 comparações. Tabela detalhada está em `relatorios/86h_hk_probe_2026-11-01-inputs/model_inputs/history_comparison.json`.
- Limite: o gabarito r2 atual vem da Passagem do Rei e não cobre os falsos blocos de HAZARD nem o gatilho INTERACTIVE da captura 86f. Para fechar essa validação faltam labels novos, com a DLL corrigida, em quadros com espinhos e banco/Elderbug; a captura manual mais recente não gerou um conjunto novo nos diretórios monitorados. O dump do colisor real também segue pendente desses objetos.

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

1. Repetir a avaliação de collider/label em espinhos e banco/Elderbug usando quadros novos após a correção e registrar o dump dos colliders. Os oito quadros do probe de política não têm suporte para essas classes.
2. A rodada longa de currículo 0–8 segue aguardando a frase de autorização do operador. Não foi iniciada.
3. Etapa 1 deverá ser repetida após a rodada longa, conforme prompt.

## Validação

- Suíte completa: 871 passaram, 4 foram pulados, 3 avisos conhecidos e 8 subtests passaram (262,63 s). Testes focados atuais do renderer/estilos: 9 passaram.
- Ruff completo: passou. Mypy: 109 arquivos sem erros.
- `git diff --check`: sem erros; Git apenas alertou sobre normalização CRLF/LF.
- Testes C# semânticos e build Release passaram antes desta última rodada; o código C# não mudou depois.
