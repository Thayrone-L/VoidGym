# 86g — ensinar a usar o ferrão

## Inventário pré-mudança

Detalhes, valores e referências de código: [`86g_inventario.md`](86g_inventario.md). A hipótese de que o shim atual não paga abates repetidos foi corrigida no inventário: `explore_v4` ainda os paga com decaimento. Os novos pagamentos são uma bonificação adicional de descoberta exclusivamente na execução v2.

## Ginásio

- A numeração passa a 0–8: o nível 6 novo ensina combate; o antigo nível 6 de quebrável passa a 7; o antigo nível 7 multi-sala passa a 8.
- A rocha de tesouro é uma entidade procedural própria, com forma facetada e quatro variações, sem arte copiada. Recebe 2–4 golpes, solta uma moeda por golpe e 2–3 no golpe final; moedas caem até o piso e só pagam ao toque. A rocha é `BREAKABLE`, igual ao GeoRock no gabarito HK; moedas são `PICKUP`.
- O nível 6 combina perigos, rochas opcionais e 1–2 inimigos de passagem em corredor com teto baixo. A colisão bloqueia até o abate; o pogo segue permitido.
- `GameRewardConfig` torna configuráveis: primeiro golpe +0,05; abate +0,3 com teto de 3 por sala; moeda +0,2. Golpes em alvo morto/rocha vazia não emitem evento.
- `solve_room(..., allow_attack=False)` falha no nível 6. Na avaliação de 200 salas, solver sem ataque falha em 200/200.

## Eventos e bônus v2 do Hollow Knight

O componente confirmado no `Assembly-CSharp` é `GeoRock`; o layout/dump da instância registra `GeoRock` em `Geo Rock 1/_Props`, e a inspeção anterior encontrou `GeoRockData` com `id`, `sceneName` e `hitsLeft`. Os eventos novos são `hit_enemy`, `kill_enemy`, `hit_geo_rock`, `geo_rock_broken` e `geo_collected`.

O bridge anuncia `combat_events_v1` e só serializa `combat_events` em FRAME quando o cliente pede essa capacidade. A captura usa IDs estáveis de hierarquia por cena; o leitor do lado Python rejeita eventos fora do contrato. O request é ativado somente pelo runtime v2. `HollowKnightAdapter` deixa o novo ledger desligado por padrão, então o shim v1 não recebe a bonificação.

`DiscoveryRewardConfig` aplica +0,05 no primeiro golpe em GeoRock, +0,1 ao quebrar e +0,15 no primeiro abate por cena/objeto durante a jogada. Os tetos iniciais são +0,5 por cena e +2,0 por fatia de episódio. O ledger preserva objetos descobertos entre reentradas; geo coletado continua sem bônus direto. Testes de unidade cobrem pagamento único, repetição, caps e opt-in.

## Frequência de ações e avaliação

Baseline: `universal-009235520`, split `test`, 200 salas por nível antigo. A coleta amostra as primeiras 40 decisões por sala para limitar uso de recursos; os IDs antigos são 0–5, 7, 8 na numeração nova. O JSON completo é `86g_action_frequency_before.json`.

Baseline concluída antes das mudanças. `attack_actions` agrega `ATTACK`, variações direcionais/pogo e `NAIL_ART`; o JSON [`86g_action_frequency_before.json`](86g_action_frequency_before.json) contém a contagem separada de cada ação. A máscara deixou um ataque disponível em **100%** das decisões medidas:

| Nível antigo | Decisões | Ações de ataque | Ataque disponível na máscara | Completaram em 40 decisões |
|---:|---:|---:|---:|---:|
| 0 | 8.000 | 463 | 100% | 0 |
| 1 | 8.000 | 13 | 100% | 0 |
| 2 | 8.000 | 2.411 | 100% | 0 |
| 3 | 8.000 | 141 | 100% | 0 |
| 4 | 8.000 | 78 | 100% | 0 |
| 5 | 8.000 | 917 | 100% | 0 |
| 6 (novo ID 7) | 8.000 | 405 | 100% | 0 |
| 7 (novo ID 8) | 7.667 | 478 | 100% | 34 |

Os 34 sucessos do multi-sala chegaram ao fim antes da janela de 40 ações; os outros conjuntos foram censurados no limite. Esta medição descreve uso inicial da política por sala/conjunto, não a taxa de conclusão do episódio. O mosaico e os GIFs dos níveis 6–8 foram regenerados e inspecionados; o GIF do nível 6 mostra o corredor sob teto baixo, e a rocha procedural tem variação facetada própria.

### Baselines determinísticas e solver

Avaliação no split de teste, 200 seeds por nível, até 300 ações por sala (600 no multi-sala); JSON completo em [`86g_politicas_bobas.json`](86g_politicas_bobas.json). Em todos os níveis 2–8, solver concluiu 200/200, e as três políticas simples falharam em pelo menos 80%:

| Nível | Solver | RIGHT falha | RIGHT+JUMP_TAP falha | RANDOM_UNIFORM falha |
|---:|---:|---:|---:|---:|
| 2 | 200/200 | 98,5% | 97,5% | 99,5% |
| 3 | 200/200 | 80,0% | 88,5% | 92,5% |
| 4 | 200/200 | 100% | 100% | 92,0% |
| 5 | 200/200 | 82,5% | 98,0% | 97,5% |
| 6 — combate | 200/200 | 100% | 100% | 99,0% |
| 7 — quebrável | 200/200 | 100% | 100% | 98,5% |
| 8 — multi-sala | 200/200 | 100% | 100% | 100% |

No nível 6, o solver com ataque desligado concluiu 0/200 (falha de 100%). No nível 8, o solver sem a habilidade exigida para a porta concluiu 0/200 conjuntos (falha de 100%).

O checkpoint depois do treino depende de autorização específica para treino longo; nenhuma campanha será iniciada sem ela. A rodada curta em `hk-ai-1`, o GIF HK de 30 s e o hash/restauração do save isolado aguardam a janela operacional que o operador disse que informará. A frequência HK antes/depois depende dessas rodadas; por enquanto só há o baseline do ginásio registrado acima.

## Validação

- Suíte Python completa: **864 passaram, 4 foram ignorados**, sem falhas; 3 avisos do SB3 sobre cópia de tensor.
- Teste da opção v2 que ativa rótulos de quebráveis no processo isolado: passou dentro da suíte.
- Ruff (`src`, `tools`, `tests`): passou. Mypy em `src` e mypy nos quatro avaliadores alterados: passou.
- `git diff HEAD~2 --check`: passou. Build Release de `HkAgentBridge.Mod` contra o Managed isolado `hk-ai-1`: 0 avisos/erros. Harneses C# de correlação/lifecycle e rasterização/classificação semântica: passaram.
- Commits e push no PR #8 concluídos: `35318c2` (ginásio/avaliadores), `ff5d711` (bridge e recompensa v2) e `22daed7` (validação final).
- Etapa 3 ainda aguarda a janela de 3–5 min que o operador disse que informará para a captura manual em `hk-ai-1`, incluindo hash e restauração do save isolado. Não implantei DLL nem abri o jogo. Treino longo e avaliação antes/depois no checkpoint novo aguardam autorização específica na hora; nenhuma campanha foi iniciada.
