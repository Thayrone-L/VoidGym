# 87r — primeira campanha v2 no Hollow Knight e comparação com v1

**Estado em 03/11/2026:** preparação parcial. A campanha longa `v2-hk-1` não foi iniciada. As fumaças com dois atores chegaram a produzir updates, mas todas falharam no guard de commit de 3 GiB; portanto a fumaça obrigatória não passou e a comparação final ainda não pode ser concluída.

## Etapa 0 — correções e proteções

- O adaptador de treino não pede rótulos de quebráveis secretos. `GeoRock`, visível ao jogador, mantém a classe de rocha de tesouro (classe 6). Os demais quebráveis são ignorados no treino; a captura completa exige a opção de diagnóstico, desligada por padrão e não exposta no painel.
- Famílias observadas no dump/captura: `GeoRock` (geo visível), `Breakable`, `BreakableObject`, `BreakableInfectedVine`, `BreakablePole`, `BreakablePoleSimple`, `BreakablePoleTopLand` e `DestructibleProp`. As famílias genéricas são ignoradas durante o treino; não há evidência para afirmar que todas são paredes secretas.
- A rota HTTP de criação de campanha rejeitava campos válidos do `CampaignSpec` v2, inclusive `engine` e `universal_checkpoint`. A allowlist do endpoint foi alinhada ao contrato e coberta por teste.
- Learner/atores HK v2 publicam telemetria e estados terminais; o learner é iniciado pelo supervisor com limite de três retomadas em 24 horas e guarda de commit de 3 GiB. A fumaça confirmou a recuperação até atingir esse limite.
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

A campanha foi cadastrada, mas continua **não iniciada**. A versão preparada mais recente é `v0003` de `v2-hk-1` em `artifacts/control-plane/campaigns/v2-hk-1/v0003.json`:

- motor v2; checkpoint inicial `E:\VoidGymRuntime\v2\pretrain\pretrain-86e-final-resume4-2026-11-03-cuda\universal-012438528`;
- dois atores e learner compartilhado; rollout de 256 passos por ator, batch 64 e checkpoint a cada 512 transições (redução feita para testar o limite de memória; ainda não validada);
- origem do save: baseline isolada copiada para os slots gerenciados, no início de `Tutorial_01`; não usa o save da campanha v1 nem o save pessoal;
- recompensa `explore_v4`, augmentação espacial, losses auxiliares v2, self-imitation, prática/desafios e pedidos de ajuda habilitados; gatilhos de ajuda: 10 travadas / 60 min;
- escala de LR do encoder: 0,3 (padrão do learner v2); taxa principal `0,00025`;
- limite planejado de 6 h. Essa duração só se aplica após a frase de autorização exigida pelo prompt.

As versões anteriores `v0001` (512/256) e `v0002` (512/128) foram preservadas no histórico; ambas também falharam no guard durante as fumaças. Não foi alterado nem reduzido o guard de 3 GiB.

### Fumaças e resultado

| Tentativa | Perfil | Resultado |
|---|---|---|
| r1 | spec inicial | Falhou antes do jogo: flags do learner foram passadas ao supervisor; 0 passos. |
| r2 | argumentos corrigidos, limite de 25.000 passos | Terminou no checkpoint inicial `12.438.528`, antes de qualquer rollout; 0 passos novos. |
| r3 | contínua, 512/256, 10 min | Duas tentativas do learner falharam quando o commit livre caiu abaixo de 3 GiB; supervisor retomou e obteve updates finitos; smoke cancelado no limite. |
| r4 | contínua, 512/256, 10 min | Duas falhas iguais; terminou em `12.440.576` (+2.048 passos); losses finitas. |
| r5 | contínua, 512/128, 10 min | Duas falhas iguais; supervisor retomou; cancelado no limite. Reduzir o batch não resolveu. |
| r6 | contínua, 256/64 | Quatro tentativas falharam (três retomadas em 24 h). O grupo encerrou como `FAILED` antes de completar 10 min, com motivo `learner restart limit reached`; no erro final havia ~1,17 GiB de commit e ~0,86 GiB de RAM livres. Terminou em ~12.440.576 passos no `latest.json`, mas o status terminal reportou 12.440.064 (diferença de 512 passos), que fica registrada como discrepância de estado. |

As losses v2 (PPO, semântica, dinâmica e self-imitation) foram finitas nos updates publicados. Os dois atores avançaram e o bridge iniciou em `Tutorial_01`. Isso valida parte do caminho de integração, mas **não** substitui o critério de fumaça sem falhas. Nenhum pedido de ajuda foi resolvido por conta própria. Todos os processos dos atores e do jogo foram fechados; runtimes e saídas temporárias das fumaças foram limpos. Os registros de auditoria e specs permanecem em `C:\VoidGym\artifacts\control-plane\`.

## Comparação, curvas e recomendação

| Métrica | v1 | v2 (`v2-hk-1`) |
|---|---:|---:|
| Progresso em 25k / 50k / 100k / 150k passos | indisponível nos resumos globais | pendente: campanha não iniciada |
| Passos até 100 / 130 unidades | episódio 15, sem coordenada global exata | pendente |
| Saiu de `Tutorial_01` | não | pendente |
| Travadas/h, golpes, abates, geo, ajuda e passos/s | dados parciais, sem pares por passo | pendente |
| IoU início / 2 h / fim | n/a | só medição inicial disponível |

Ainda não há dados para recomendar a v2 como padrão nem para gerar curvas/GIFs comparáveis. A recomendação fica pendente; não extrapolar os dois updates da fumaça para desempenho de campanha.

## Validação

Validação local concluída:

- Suíte unitária: 880 passaram, 4 ignorados, 2 desmarcados e 8 subtestes passaram.
- Suíte E2E: 2 passaram; 884 testes não E2E foram desmarcados nessa execução.
- Ruff e mypy passaram (111 arquivos-fonte, sem erros).
- Build do bridge: sucesso, zero avisos e zero erros; testes de contrato semântico passaram.
- `git diff --check` passou.
- As fumaças continuam reprovadas pelo guard de commit; portanto não aprovam a campanha.

O CI remoto apontou falha inicial de imports e dependências ausentes. Os pacotes locais `tests` e `tests.v2` agora têm `__init__.py`; os locks de treino, mapa e desenvolvimento são instalados nos jobs que precisam deles. O mypy completo roda no job Windows v2, e a suíte Python completa também foi movida para Windows porque seus testes de instalação e diretórios dependem de caminhos e variáveis desse sistema. O check separado de fronteira portátil continua em Linux. Após esses ajustes, os cinco checks remotos passaram no commit `5af141a`: Python, bridge contracts, portable policy boundary e os dois eventos do workflow v2 (push e pull request).

## Próxima ação necessária

É preciso criar headroom persistente para o commit de sistema permanecer acima de 3 GiB durante o primeiro update com dois jogos e o learner. Fechar apps não usados e liberar memória antes de outra fumaça é a opção reversível. Enquanto esse teste não passar, não iniciar a campanha longa. Após uma fumaça aprovada, pedir a frase literal `autorizo iniciar a campanha 87` antes de qualquer execução de até 6 h.
