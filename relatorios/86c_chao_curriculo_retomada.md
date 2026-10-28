# 86c — chão, currículo e retomada

## Estado e escopo

Implementadas as correções de runtime, currículo, gerador e rasterizador sem alterar `src/hk_agent/` (v1). A retomada longa permanece pendente de autorização nova na hora, como exige o prompt; nenhum treino foi iniciado ou parado durante esta etapa. A campanha de 86b já havia encerrado em 27/10 por falha do worker, antes destas mudanças.

## Diagnóstico e salvaguardas do pré-treino

O último registro de `pretrain-86b-gpu-2026-10-27` foi às 16:01:47 -03, em 4.526.848 passos totais (3.829.760 efetivos desde 697.088), aproximadamente 4h40 após o início. O checkpoint íntegro mais recente é `universal-004460288`; os 66.560 passos posteriores aparecem apenas no CSV. Às 16:01:53, o Windows registrou Application Error 1000 para o worker Python (PID 19672), módulo `VCRUNTIME140.dll`, exceção `0xc0000005`; WER 1001 veio às 16:01:58. Não houve eventos System 41, 6008, 1074 ou 4101 no intervalo. Isso aponta para falha do processo, mas não identifica a causa interna.

`tools/v2/pretrain_platform.py` agora grava heartbeat de passos/nível/horário, estados terminais atômicos (`COMPLETED`, `STOPPED`, `FAILED`), traceback em stderr, alerta se uma execução anterior deixou heartbeat sem status, e fecha recursos/restaura handlers mesmo em falha. O teste de falha forçada confirma status FAILED e traceback. A avaliação rápida é configurável; o teste curto com intervalo de 5.000 em treino de 12.000 verifica valores no CSV. A premissa original do 86c sobre avaliação ausente estava incorreta: o CSV da campanha traz quatro avaliações no intervalo de 1M (1.001.728: 0,3571; 2.000.128: 0,7143; 3.001.088: 0,7143; 4.002.048: 0,5).

## Geometria e políticas simples

O gerador limita obstáculos obrigatórios a 85% da capacidade física. Nível 2 usa vão 0,35–0,70 do alcance e barreira 0,55–0,75 da altura máxima, com pelo menos 3 unidades de corrida. Níveis 4/5 ficam abaixo do teto de 0,85. A suíte do gerador verificou solvabilidade e desempenho; teste final: 6 passaram.

Avaliação de 200 mapas por nível, solver e políticas simples (falha):

| Nível | 86b: RIGHT / toque / aleatória | 86c: RIGHT / toque / aleatória | Solver 86c |
|---:|---:|---:|---:|
| 2 | 1,00 / 0,99 / 0,99 | 0,98 / 0,95 / 0,99 | 1,00 |
| 3 | 0,835 / 0,905 / 0,91 | 0,835 / 0,905 / 0,91 | 1,00 |
| 4 | 1,00 / 1,00 / 0,80 | 1,00 / 1,00 / 0,81 | 1,00 |
| 5 | 0,85 / 0,955 / 0,99 | 0,845 / 0,955 / 0,99 | 1,00 |
| 6 | 1,00 / 1,00 / 0,995 | 1,00 / 1,00 / 0,995 | 1,00 |
| 7 | 1,00 / 1,00 / 1,00 | 1,00 / 1,00 / 1,00 | 1,00 |

Cada política falha em pelo menos 80% por nível; a porta do nível 7 sem habilidade falha 200/200. Dados completos: `86c_politicas_bobas.json`. A tabela anterior está em `86b_gabarito_e_pre-treino.md`.

O currículo registra histórico por nível, preserva/migra o checkpoint antigo e sorteia 60% nível atual, 30% níveis inferiores e 10% nível seguinte (no nível 0, 90% atual e 10% seguinte). Há teste de agente com 60% no nível 2 ainda amostrar nível 3. O log passa a incluir sucesso por nível. Não existe ainda série temporal pós-mudança nem PNG de evolução: a campanha não foi retomada. O CSV antigo registra somente nível atual e, portanto, não permite reconstruir métricas por nível sem inventá-las.

## Gabarito semântico e captura em movimento

`EdgeCollider2D` fechado é preenchido como polígono; linha aberta é SOLID e o lado de terreno atrás dela é IGNORE até outra linha ou borda. Testes C# cobrem chão aberto e contorno fechado preenchido. INTERACTIVE agora depende de componente; `RestBench`, `BridgeLever`, `BossStatueLever` e `StagTravel` são suportados no assembly da instância. A camada sozinha não basta; grama/cortáveis ficam EMPTY sem colisão/dano relevante. Detalhes e limites de componentes estão em `docs/v2/hk-semantic-terrain.md`.

A amostra consolidada tem 1.500 quadros do `Tutorial_01`: 36 capturas guiadas variadas e 1.464 quadros uniformemente amostrados da rota automatizada; 1.499 têm ação associada. A rota automatizada repetia perto do vão inicial. O roteiro guiado alcançou o Crawlid e o trecho inferior, mas o personagem morreu antes de transição de cena. Um portão/estrutura aparece no extremo direito, mas não foi confirmado como saída. As máscaras não marcaram HAZARD nem EXIT: as frações dessas classes são zero, apesar de aparecerem elementos visualmente semelhantes a espinhos e uma estrutura candidata a portão. Portanto, requisito de capturar/rotular espinhos e saída continua incompleto. A amostra inclui EMPTY 0,243645; SOLID 0,354570; ENEMY 0,397170; PLAYER 0,004615; demais classes 0. Gabarito de oito quadros anotados: `86c_hk_gabarito.png`.

O save isolado de `hk-ai-1` foi restaurado e verificado por hash: 90 arquivos, sem ausências, alterações ou extras. Save pessoal ficou isolado e intocado.

## Transferência seca

Ambos os checkpoints foram avaliados nos mesmos 1.500 quadros de `Tutorial_01`, sem treino e sem histórico de ações (manifesto marca `action_history_available:false`). IoU por classe:

| Classe | `universal-000697088` | `universal-004460288` |
|---|---:|---:|
| EMPTY | 0,2203 | 0,2108 |
| SOLID | 0,0169 | 0,0159 |
| PLATFORM | 0,0000 | 0,0000 |
| HAZARD | não avaliável | não avaliável |
| ENEMY | 0,0000 | 0,0000 |
| PLAYER | 0,1885 | 0,0572 |
| EXIT | 0,0000 | 0,0000 |

BREAKABLE, PICKUP e INTERACTIVE não têm classes positivas na amostra e ficam sem IoU definido. HAZARD também não é avaliável. As métricas e matrizes completas estão em `86c_zero_shot_000697088/` e `86c_zero_shot_004460288/`. A baixa IoU de SOLID evidencia que a transferência ainda não está boa; estes são resultados secos, não avaliação após treino.

## Validação, build e deployment

O pré-treino não foi retomado, logo não há avaliação final de política aprendida nem comparação com o split do 85. Nenhum treinamento foi iniciado ou interrompido por este trabalho.

Testes C# do rasterizador e build Release do mod passaram. DLL implantada pelo `tools/deploy_bridge.py` somente na instância isolada `hk-ai-1`; backup: `artifacts/bridge-deploy-backups/bridge-20261028-105213-8bfb98e7/hk-ai-1`. Suíte completa: 843 passaram, 4 ignorados, 8 subtests passaram em 264,79 s; três avisos existentes de conversão de tensor no Stable-Baselines3. Ruff: `All checks passed`. Mypy: sem erros nos quatro arquivos Python verificados (curriculum, generator, pretrain_platform e importador). `git diff --check`: passou. Testes C# semânticos passaram; build Release passou com zero avisos e zero erros.

## Pendências

1. Retomar de `universal-004460288` somente após autorização fresca do operador no momento do início; monitorar heartbeat e parar/investigar se um nível ficar sem progresso por 3M de passos.
2. Fazer avaliação por nível ao longo do novo treino, curva PNG, avaliação no split de teste e IoU do ginásio.
3. Capturar novamente uma amostra que alcance e rotule explicitamente espinhos e saída, sem sacrificar a restauração do save.

