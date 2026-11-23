# 86i — nível 2 e avaliação final

**Checkpoint avaliado:** `universal-012438528` (12.438.528 passos treinados)
**Campanha:** encerrada em 03/11/2026 às 13:25, `COMPLETED` pelo limite de 3 milhões de passos sem melhora no nível 2.
**Escopo executado:** avaliações offline e probe de Hollow Knight sem treinamento. Não houve alteração do gerador, currículo, checkpoint ou campanha.

## Conclusão executiva

O nível 2 não está demonstrado como impossível para as ações da IA. O solver livre resolveu 200/200 salas e a busca pela tabela discreta encontrou rotas em 97/200; as outras 103 terminaram por `node_limit`, portanto são desconhecidas, não falhas provadas. Entre as salas em que encontramos uma rota por ações, a política ainda falhou em 48/97 deterministicamente e em 36/97 estocasticamente. Isso é evidência direta de uma lacuna de seleção/execução de ações. Os GIFs mostram erros de aproximação e reversão perto do vão. As salas que a busca não resolveu têm distribuições de vão e barreira sobrepostas às resolvidas. Não há evidência suficiente para mudar faixas do gerador ou certificar novas faixas; mantivemos o gerador intacto e não retomamos treino.

## Etapa 1 — solucionabilidade e política

Busca pela tabela universal discreta, mesma tabela e durações do adaptador, 200 salas por nível, sementes iniciadas em 8600, limite de 5.000 expansões. `node_limit` significa inconclusivo.

| Nível | Solver livre | Solver por ações | Política determinística | Política estocástica |
|---:|---:|---:|---:|---:|
| 0 | 200/200 | 200/200 | 99,0% | 96,0% |
| 1 | 200/200 | 200/200 | 67,0% | 100,0% |
| 2 | 200/200 | 97/200; 103 `node_limit` | 50,5% | 60,5% |
| 3 | 200/200 | 198/200; 2 `node_limit` | 90,5% | 99,0% |
| 4 | 200/200 | 200/200 | 46,5% | 95,5% |
| 5 | 200/200 | 194/200; 6 `node_limit` | 76,5% | 94,0% |
| 6 | 200/200 | 200/200 | 97,0% | 99,5% |
| 7 | 200/200 | 192/200; 8 `node_limit` | 79,5% | 93,5% |
| 8 | 200/200 | 164/200; 36 `node_limit` | 35,0% | 71,0% |

No cruzamento individual do nível 2, o solver por ações encontrou rota em 49 salas nas quais a política determinística teve sucesso e em 48 nas quais falhou. Outras 52 salas sem rota encontrada tiveram sucesso determinístico e 51 falha. No modo estocástico, as contagens foram 61 sucessos e 36 falhas entre as salas resolvidas, contra 60 sucessos e 43 falhas entre as salas `node_limit`. A busca limitada não separa as geometrias: média de vão/alcance 54,15% nas salas resolvidas contra 51,17% nas inconclusivas; barreira/altura máxima de salto 64,65% contra 64,96%. Assim, não tratamos os 103 casos como impossíveis.

As políticas de controle simples falharam em pelo menos 80% das salas dos níveis 2–8. No nível 2, as taxas de falha foram 98,5% para `RIGHT`, 97,0% para `RIGHT+JUMP_TAP` e 97,5% para ações aleatórias uniformes. Resultados por nível constam em [`dumb_policies.json`](86i_nivel2_e_avaliacao/dumb_policies.json).

### Evidência de comportamento no vão

No teste completo de 200 episódios por modo/nível, nível 2 teve 101/200 sucessos determinísticos (50,5%) e 121/200 estocásticos (60,5%). A política determinística escolheu `LEFT+ATTACK` 58.038 vezes e `ATTACK` 41.202 vezes no nível 2; a estocástica escolheu `ATTACK` 39.839, `LEFT+ATTACK` 24.407, `JUMP_TAP` 17.461 e `LEFT+JUMP_TAP` 14.852 vezes. Na amostra de trace perto do vão (60 episódios determinísticos), predominaram DOWN+ATTACK (341), LEFT+ATTACK (197), LEFT+JUMP_TAP (138), JUMP+ATTACK (130), RIGHT+ATTACK (97) e RIGHT+JUMP_MAX (75). A primeira falha inspecionada (`seed=28600`) oscila de avanço para reversão junto ao vão. O contato visual dos dez episódios está em [`l2_failure_contact.png`](86i_nivel2_e_avaliacao/behavior_l2_trace/l2_failure_contact.png); os traces brutos permanecem nos resultados locais.

Não aplicamos mudança de recompensa, peso de currículo nem término de episódio: a evidência mostra erro de política, mas esta avaliação não isolou qual intervenção corrige a causa. Qualquer alteração e novo treino exigem decisão/autorização futura. O limite de busca de 20.000 expansões foi iniciado como retry opcional, mas não concluiu nenhum dos 103 casos antes de ser interrompido para evitar pressão de memória; o arquivo de progresso não representa uma rodada concluída.

### Comparação com a tabela do 86e

Resultados históricos arquivados do 86e, por nível 0–7: determinístico 93%, 85%, 54%, 86%, 76%, 95%, 5%; estocástico 93%, 99%, 59%, 99%, 90%, 98%, 94%. A coleta histórica tinha 100 episódios nos níveis 0–6 e 30 no nível 7, sem nível 8. A rodada 86i usou 200 em cada nível 0–8, logo diferenças pequenas não devem ser tratadas como comparação pareada. O nível 2 continua próximo ao resultado histórico; a nova amostra e o solver de ações dão evidência adicional sobre seleção de ação, não uma correção validada.

## Etapa 3 — comportamento completo

Resultados de `universal-012438528`, 200 episódios por nível e modo, em divisão de teste:

| Nível | Determinístico | Estocástico |
|---:|---:|---:|
| 0 | 99,0% | 96,0% |
| 1 | 67,0% | 100,0% |
| 2 | 50,5% | 60,5% |
| 3 | 90,5% | 99,0% |
| 4 | 46,5% | 95,5% |
| 5 | 76,5% | 94,0% |
| 6 | 97,0% | 99,5% |
| 7 | 79,5% | 93,5% |
| 8 | 35,0% | 71,0% |

As frequências completas de ação e passos/mortes/travadas por nível estão em [`behavior_test/tabela.md`](86i_nivel2_e_avaliacao/behavior_test/tabela.md) e [`behavior_progress.json`](86i_nivel2_e_avaliacao/behavior_test/behavior_progress.json). Como referência histórica, o 86e tinha deterministicamente 93/85/54/86/76/95/5% e estocasticamente 93/99/59/99/90/98/94% nos níveis 0–7, com tamanhos amostrais diferentes indicados acima. As três políticas simples e seus resultados estão em [`dumb_policies.json`](86i_nivel2_e_avaliacao/dumb_policies.json).

## Percepção balanceada

Foram avaliados 625 quadros por nível em cada split (5.625 por split, 100 episódios por nível). Os splits agora usam sementes de geometria disjuntas: train começa em 1.085.000 e test aplica offset de 1.000.000. Uma coleta preliminar tinha reutilizado as mesmas geometrias nos dois splits; ela foi descartada e não embasa as métricas abaixo. Os manifestos registram hash dos IDs amostrados.

| Classe | Treino | Teste |
|---|---:|---:|
| EMPTY | 0,956 | 0,954 |
| SOLID | 0,445 | 0,435 |
| PLATFORM | 0,158 | 0,154 |
| HAZARD | 0,179 | 0,173 |
| ENEMY | 0,107 | 0,101 |
| PLAYER | 0,707 | 0,605 |
| BREAKABLE | 0,251 | 0,221 |
| PICKUP | 0,000 | 0,000 |
| INTERACTIVE | 0,019 | 0,028 |
| EXIT | 0,292 | 0,288 |

O IoU de teste é baixo para quase todas as classes semânticas, apesar de `L_semantic=0,0199`; portanto a perda baixa não equivale a boa percepção. A transferência melhora em relação ao checkpoint pareado `universal-009235520` em todos os IoUs medidos, mas ainda é fraca (por exemplo, SOLID 0,285→0,435, HAZARD 0,009→0,173, PLAYER 0,526→0,605). Os quadros dos dois checkpoints têm os mesmos IDs/hash de amostra.

No teste, silhueta (123 quadros por nível) versus demais estilos (502 por nível): SOLID 0,402 vs 0,444; PLAYER 0,636 vs 0,598; HAZARD 0,171 vs 0,173; ENEMY 0,124 vs 0,097; EXIT 0,297 vs 0,286. Não aparece uma penalidade geral da silhueta; SOLID é menor nesse grupo, enquanto PLAYER/ENEMY/EXIT são semelhantes ou um pouco maiores. O treino tem 135 silhuetas e 490 outros quadros por nível; as métricas por estilo, suporte e matrizes estão nos diretórios `perception_train_disjoint` e `perception_test_disjoint`.

A comparação com a tabela antiga do 86f não é pareada: o relatório antigo cobria 5.000 quadros, níveis 0–7, e uma avaliação que não garantia geometrias disjuntas. Não use a diferença como medida limpa de progresso. A rodada corrigida com 86f e a comparação pareada do checkpoint antigo estão preservadas em `perception_test_disjoint`.

## Hollow Knight sem treinamento

### Recaptura corrigida

A recaptura `session-20261101-193731` contém 727 quadros; foram excluídos 21 de menu e 11 quase pretos, restando 695. A captura não salva histórico de ações, então o `action_history` de entrada foi preenchido com zeros. IoU: EMPTY 0,495; SOLID 0,084; PLATFORM 0,000; HAZARD 0,007; ENEMY 0,003; PLAYER 0,046; BREAKABLE 0,000; INTERACTIVE 0,006; EXIT 0,002. Em pixels com rótulo SOLID, 84,60% foram previstos como EMPTY (325.017 de 384.184 células). O modelo transfere muito mal para essa recaptura; a ausência de histórico de ação é uma limitação adicional.

### Probe online da política

Três inicializações do probe de 86i no slot isolado `hk-ai-1` falharam antes da primeira decisão, durante `env.reset`, com `TimeoutError` aguardando conexão do bridge. Assim não há métricas 86i válidas de x/y, direção, avanço, golpes, `hit_enemy` ou GIF de 30 segundos; os diretórios de tentativas contêm apenas preparação/erro e os registros de restauração. A janela do jogo mostrou o mod `HkAgentBridgeMod` carregado, mas isso não bastou para o supervisor estabelecer a conexão. A execução online anterior do 86h (checkpoint `universal-009235520`) durou 304,4 s, fez 1.044 decisões e gerou GIF, porém precede a alteração de instrumentação e não contém a telemetria x/y e `hit_enemy` pedida para 86i; não substitui o probe deste checkpoint.

Os três `save-restoration.json` das tentativas 86i confirmam `restored: true`; os hashes antes/depois coincidem e nenhuma gravação foi necessária. O save pessoal e o baseline não foram usados. O probe online permanece pendente de corrigir/investigar o timeout de conexão.

## Arquivos, código e validação

O avaliador de solvabilidade por ações e seus testes foram implementados no commit `3a52738` (`Add discrete action solvability evaluator`). Ferramentas 86i ajustam a coleta para splits de geometria realmente disjuntos, baselines bobas, traces do probe e recaptura HK. Nenhuma alteração foi feita no gerador, currículo, treino ou checkpoints.

Resultados e evidências reproduzíveis:

- [`action_solver_5k/action_solvability.json`](86i_nivel2_e_avaliacao/action_solver_5k/action_solvability.json)
- [`dumb_policies.json`](86i_nivel2_e_avaliacao/dumb_policies.json)
- [`behavior_test/tabela.md`](86i_nivel2_e_avaliacao/behavior_test/tabela.md)
- Manifestos e matrizes em `perception_train_disjoint`, `perception_test_disjoint` e `hk_recapture`.
- Contact sheet dos traces em [`l2_failure_contact.png`](86i_nivel2_e_avaliacao/behavior_l2_trace/l2_failure_contact.png).

Validação executada em 03/11/2026:

- Suíte completa: **870 passed, 4 skipped, 3 failed**, 8 subtests passaram. As três falhas estão em `tests/test_control_plane.py` (`test_parallel_disabled_blocks_second_real_launch` e duas variantes de `test_final_instance_saves_are_captured_on_failure_and_stop`): o `psutil.Process(pid).ppid()` observa `NoSuchProcess` porque os PIDs falsos/efêmeros já desapareceram. Não são falhas dos módulos do 86i.
- Teste dirigido `tests/test_platform_action_solver.py`: **2 passed**.
- `ruff check .`: **All checks passed**.
- Mypy estrito dos sete módulos 86i com imports indiretos silenciosos e stubs ausentes ignorados: **Success: no issues found in 7 source files**.
- Mypy global: não fecha no estado atual do repositório. Sem filtro, a descoberta inclui diretórios de backup com nomes inválidos e módulos duplicados; excluindo esses backups e usando bases explícitas, ainda encontra vários erros e imports sem stubs em código existente fora do escopo do 86i.
- `git diff --check`: passou; apenas avisos do Git sobre normalização CRLF/LF em dois arquivos existentes.
