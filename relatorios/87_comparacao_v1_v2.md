# 87r — primeira campanha v2 no Hollow Knight e comparação com v1

**Estado em 04/11/2026, 16:35 BRT:** a fumaça 87r de 10 min passou com 2 atores e perdas auxiliares ligadas. A campanha longa `v2-hk-1`, versão 4, foi iniciada às 16:34 BRT a partir de `universal-012438528`; comparação final pendente.

## Etapa 0 — correções e proteções

- O adaptador de treino não pede rótulos de quebráveis secretos. `GeoRock`, visível ao jogador, mantém a classe de rocha de tesouro (classe 6). Os demais quebráveis são ignorados no treino; a captura completa exige a opção de diagnóstico, desligada por padrão e não exposta no painel.
- Famílias observadas no dump/captura: `GeoRock` (geo visível), `Breakable`, `BreakableObject`, `BreakableInfectedVine`, `BreakablePole`, `BreakablePoleSimple`, `BreakablePoleTopLand` e `DestructibleProp`. As famílias genéricas são ignoradas durante o treino; não há evidência para afirmar que todas são paredes secretas.
- A rota HTTP de criação de campanha rejeitava campos válidos do `CampaignSpec` v2, inclusive `engine` e `universal_checkpoint`. A allowlist do endpoint foi alinhada ao contrato e coberta por teste.
- Learner/atores HK v2 publicam telemetria e estados terminais; o learner é iniciado pelo supervisor com limite de três retomadas em 24 horas e guarda de commit de 3 GiB. A fumaça confirmou a recuperação até atingir esse limite.
- `aux_losses` agora funciona de ponta a ponta: o painel aceita a opção, inclui `--aux-losses` no comando do learner e o learner só passa os coeficientes de semântica/dinâmica quando a flag está ligada. `asymmetric_critic` e `mask_interact_without_prompt` continuam bloqueados.
- A DLL corrigida foi implantada somente por `tools/deploy_bridge.py` nas instâncias `hk-ai-1` e `hk-ai-2`; os dois hashes implantados coincidiram. Os saves das duas instâncias foram restaurados/verificados por SHA-256 contra os backups após cada fumaça. Não houve processo do jogo pessoal nem Steam.

## Etapa 1 — linha de base v1

Fonte: `jogada-1-repeat-1c3b` e seus manifestos, resumos de episódio e avaliações em `C:\Users\thayr\Games\HollowKnight-AI\runtime\shared`. A v1 permanece parada.

| Medida | v1 (`jogada-1-repeat-1c3b`) |
|---|---:|
| Passos globais | 146.944 |
| Rodadas | 155 |
| Relógio total | 11,19 h |
| Tempo de atualização do learner | 2.284 s (~38,1 min) |
| Maior progresso | 131,37 unidades em `Tutorial_01` (Passagem do Rei) |
| Transição para outra cena | Não observada |
| Encerramento | Disco cheio, em 21/10 |

O maior progresso passou de 100 e 130 unidades no episódio 15. Há 51 resumos, dos quais 49 têm progresso numérico; 46 dos 51 foram marcados como travados. Esses resumos não guardam o passo global correspondente. Por isso, [87_progresso_v1_por_passos.png](87_progresso_v1_por_passos.png) usa a soma dos passos locais dos episódios como eixo aproximado e mostra o total global de 146.944 separadamente; não deve ser lido como uma série global exata.

## Percepção inicial

Foram calculados IoU com os dois conjuntos fixos do prompt. O conjunto manual excluiu 40 quadros `Menu_Title` e avaliou os 1.532 restantes. Os valores baixos reforçam que a campanha começa a partir de uma percepção fraca:

| Classe | Recaptura corrigida | Captura manual |
|---|---:|---:|
| EMPTY | 0,4946 | 0,4691 |
| SOLID | 0,0839 | 0,0865 |
| PLATFORM | 0,0000 | 0,0000 |
| HAZARD | 0,0073 | 0,0075 |
| ENEMY | 0,0027 | 0,0026 |
| PLAYER | 0,0458 | 0,0448 |
| INTERACTIVE | 0,0055 | 0,0368 |
| EXIT | 0,0019 | 0,0007 |
| BREAKABLE | 0,0000 | 0,0000 |
| PICKUP | sem amostras | sem amostras |

Recaptura: 727 quadros, 21 menus e 11 quadros pretos excluídos; SHA-256 `5ae0fc1b928f30bb17700c7bb576b500fb67e6af161ac57a2edd243704816ff0`. Manual: manifesto com 1.572 quadros; SHA-256 `b0fcec2f8d79e5fd7a18c2ed33cbb65356c73d3bdf3ac9585b4c97d2158f68bb`. Métricas completas e parâmetros estão em `relatorios/87_perception_initial/`.

## Etapa 2 — configuração preparada

A configuração validada é `v0004` de `v2-hk-1` em `artifacts/control-plane/campaigns/v2-hk-1/v0004.json`; a versão está ativa:

- motor v2; checkpoint inicial `E:\VoidGymRuntime\v2\pretrain\pretrain-86e-final-resume4-2026-11-03-cuda\universal-012438528`;
- dois atores e learner compartilhado; rollout de 256 passos por ator, batch 64 e checkpoint a cada 512 transições (redução feita para testar o limite de memória; ainda não validada);
- origem do save: baseline isolada copiada para os slots gerenciados, no início de `Tutorial_01`; não usa o save da campanha v1 nem o save pessoal;
- recompensa `explore_v4`, augmentação espacial, perdas auxiliares v2, self-imitation, prática/desafios e pedidos de ajuda habilitados; gatilhos de ajuda: 10 travadas / 60 min;
- escala de LR do encoder: 0,3 (padrão do learner v2); taxa principal `0,00025`;
- limite planejado de 6 h. Essa duração só se aplica após a frase de autorização exigida pelo prompt.

As versões anteriores `v0001` (512/256), `v0002` (512/128) e `v0003` (256/64, sem perdas auxiliares) foram preservadas no histórico. Não foi alterado nem reduzido o guard de 3 GiB.

### Fumaças e resultado

| Tentativa | Perfil | Resultado |
|---|---|---|
| r1 | spec inicial | Falhou antes do jogo: flags do learner foram passadas ao supervisor; 0 passos. |
| r2 | argumentos corrigidos, limite de 25.000 passos | Terminou no checkpoint inicial `12.438.528`, antes de qualquer rollout; 0 passos novos. |
| r3 | contínua, 512/256, 10 min | Duas tentativas do learner falharam quando o commit livre caiu abaixo de 3 GiB; supervisor retomou e obteve updates finitos; smoke cancelado no limite. |
| r4 | contínua, 512/256, 10 min | Duas falhas iguais; terminou em `12.440.576` (+2.048 passos); losses finitas. |
| r5 | contínua, 512/128, 10 min | Duas falhas iguais; supervisor retomou; cancelado no limite. Reduzir o batch não resolveu. |
| r6 | contínua, 256/64 | Quatro tentativas falharam (três retomadas em 24 h). O grupo encerrou como `FAILED` antes de completar 10 min, com motivo `learner restart limit reached`; no erro final havia ~1,17 GiB de commit e ~0,86 GiB de RAM livres. Terminou em ~12.440.576 passos no `latest.json`, mas o status terminal reportou 12.440.064 (diferença de 512 passos), que fica registrada como discrepância de estado. |
| r7 | versão isolada, 256/64, `aux_losses=true`, 10 min | Aprovada: encerrou como `CANCELLED` por `time_limit`, `safe=true`, 4 rodadas, 2.048 passos novos (de 12.438.528 a 12.440.576), nenhum erro/retomada; losses PPO, semântica, dinâmica e total finitas. RAM livre mínima ~2,04 GiB; commit atual mínimo ~2,25 GiB e potencial ~18,25 GiB (guard 87t: atual >1,5 GiB, potencial >3 GiB). `active_count=0`, nenhum pedido de ajuda e restauração dos saves concluída. Relatório: `artifacts/reports/v2-hk-1-smoke-87r-aux-20261104/relatorio-20261104-163319-784161500.json`. |

Na r7, as losses v2 (PPO, semântica, dinâmica e total) foram finitas em todos os updates. Os dois atores avançaram e o bridge iniciou em `Tutorial_01`. O smoke terminou limpo por tempo; os processos foram encerrados e os saves restaurados. Nenhum pedido de ajuda foi resolvido por conta própria. Os registros e specs das fumaças permanecem em `C:\VoidGym\artifacts\control-plane\`.

### Campanha principal — início

Às 16:34 BRT de 04/11, `v2-hk-1` v0004 iniciou com 2 atores, rollout/batch 256/64, perdas auxiliares habilitadas e duração máxima de 6 h. Checkpoint inicial: `universal-012438528`. No primeiro minuto, learner/atores/jogos estavam ativos, com RAM livre ~3,83 GiB e commit atual/potencial livres ~8,23/~24,23 GiB. A telemetria ainda mostrava rodada 0 durante carregamento; continuar registrando por hora e medir IoU a cada 2 h. Deixar pedidos de ajuda abertos para o operador.

## Comparação, curvas e recomendação

| Métrica | v1 | v2 (`v2-hk-1`) |
|---|---:|---:|
| Progresso em 25k / 50k / 100k / 150k passos | indisponível nos resumos globais | pendente: campanha não iniciada |
| Passos até 100 / 130 unidades | episódio 15, sem coordenada global exata | pendente |
| Saiu de `Tutorial_01` | não | pendente |
| Travadas/h, golpes, abates, geo, ajuda e passos/s | dados parciais, sem pares por passo | pendente |
| IoU início / 2 h / fim | n/a | só medição inicial disponível |

Ainda não há dados para recomendar a v2 como padrão nem para gerar curvas/GIFs comparáveis. A recomendação fica pendente; não extrapolar os quatro updates da fumaça para desempenho de campanha.

## Validação

Validação local (histórico) e alteração recente:

- Suíte unitária: 880 passaram, 4 ignorados, 2 desmarcados e 8 subtestes passaram.
- Suíte E2E: 2 passaram; 884 testes não E2E foram desmarcados nessa execução.
- Ruff e mypy passaram (111 arquivos-fonte, sem erros).
- Build do bridge: sucesso, zero avisos e zero erros; testes de contrato semântico passaram.
- `git diff --check` passou.
- Após implementar `aux_losses`, `tests/test_control_plane.py`: 97 passaram; Ruff e `git diff --check` nos arquivos alterados passaram.
- As fumaças continuam reprovadas pelo guard de commit; portanto não aprovam a campanha.

O CI remoto apontou falha inicial de imports e dependências ausentes. Os pacotes locais `tests` e `tests.v2` agora têm `__init__.py`; os locks de treino, mapa e desenvolvimento são instalados nos jobs que precisam deles. O mypy completo roda no job Windows v2, e a suíte Python completa também foi movida para Windows porque seus testes de instalação e diretórios dependem de caminhos e variáveis desse sistema. O check separado de fronteira portátil continua em Linux. Após esses ajustes, os cinco checks remotos passaram no commit `5af141a`: Python, bridge contracts, portable policy boundary e os dois eventos do workflow v2 (push e pull request).

## Próxima ação

Acompanhar `v2-hk-1` v0004 por até 6 h, registrar métricas por hora e IoU a cada 2 h. Não resolver pedidos de ajuda sem o operador. Depois atualizar comparação, curvas e recomendação; não inferir desempenho da campanha com base no smoke.

### Acompanhamento da campanha longa — marcos de 30/60+ min

- **+65 min (17:39 BRT, 04/11):** a API do painel reportou `active_count=3`, com learner e dois atores em `RUNNING`; processos dos dois jogos também estavam vivos. Na visão agregada do operador, `training_steps=0`, `learner_updates=0` e `steps_per_second=0`, sem dados de cena/progresso. Nenhuma falha ou `heartbeat expirou` foi reportada. Esse retrato agregado era inconclusivo quanto ao avanço real.
- **+83,9 min (17:57 BRT, 04/11):** learner e dois atores continuavam `RUNNING`, com processos dos dois jogos vivos e sem falha/expiração de heartbeat reportada. O estado compartilhado do learner chegou à rodada 46 e `12.462.080` passos, ou **+23.552** desde o checkpoint inicial `12.438.528`. RAM disponível: ~2,43 GiB; commit livre atual: ~2,25 GiB; commit potencial com crescimento de paginação: ~18,23 GiB. O operador ainda mostra `0` updates e taxa de passos nula, e as métricas de velocidade dos atores estão indisponíveis; há, portanto, divergência entre o resumo do operador e o estado compartilhado do learner. Três pedidos de ajuda permanecem pendentes; aguardam o operador.

A verificação usou somente a API local de status e a enumeração de processos, sem abrir os arquivos do runtime.
