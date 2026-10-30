# 86g — inventário pré-mudança

Estado: inspeção estática concluída; frequência do ginásio em execução com `universal-009235520`; rodada curta HK pendente de janela do operador para `hk-ai-1`.

## Ginásio

O gerador anterior tinha níveis 0–7. Os níveis 5–7 geravam inimigos (`src/voidgym/gyms/platform_gym/generator.py:130–131`); o nível 6 também exigia quebrar uma barreira (antes da renumeração, correspondia ao atual nível 7; a certificação está em `generator.py:243–244`). O nível 7 anterior era uma sequência de 3–6 salas com porta que exige `ability_double_jump` (a função continua em `generator.py:249` e agora é selecionada no nível 8 em `adapter.py:53,72`). Nenhuma sala exigia atingir um inimigo. Os inimigos eram evitáveis e não bloqueavam movimento.

Recompensas anteriores em `src/voidgym/gyms/platform_gym/reward.py` (`GameReward` e `CombinedPlatformReward`, agora nas linhas 30 e 75):

| Evento | Recompensa atual | Limite |
|---|---:|---|
| saída | +1,0 | por evento |
| moeda | +0,2 | por coleta |
| habilidade | +1,0 | por coleta |
| banco | +0,5 | uma vez por sala/alvo |
| quebrável (`break`) | +0,3 | uma vez por sala/alvo |
| abate (`kill`) | +0,1 | até 3 por sala |
| dano | −0,3 | por ponto, via `CombinedPlatformReward`/`RewardConfig` |
| golpe sem abate | 0 | não existia evento/recompensa de hit |

## Hollow Knight v2 e shim `explore_v4`

`src/voidgym/adapters/hollow_knight/runtime.py:26,54` exige `reward_version=explore_v4`, constrói o `ExplorationTaskConfig` e fixa `geo_gain_reward=0.0` e `geo_spend_reward=0.0`. Portanto, geo ganho/gasto é observado, mas não gera recompensa direta no shim v2.

Em `src/hk_agent/exploration_task.py:68,236–237,1171–1226`, `explore_v4` desliga o componente genérico de abate e aplica eventos tipados de abate confirmado. A recompensa-base configurada é +1,0, com orçamento de +5,0 por episódio (`enemy_kill_reward=1.0`, `enemy_kill_reward_budget=5.0`). Abates repetidos ainda recebem recompensa decrescente: jogada contínua `1/(1+visitas)^2`; não contínua `max(0,2, 1/sqrt(1+visitas))`, limitado pelos tetos existentes. Golpes em inimigos não têm recompensa dedicada. A recompensa por golpe em depósito de geo também não existe no shim atual. Os campos de geo existentes (`supervisor.py:61–64,396–397`) não são monetizados no v2.

Observação: a hipótese do texto do 86g de que o shim atual paga zero por abates repetidos não corresponde ao código. A mudança da etapa 2 deve adicionar apenas a recompensa de descoberta nova, sem editar `combat_reward.py`/`explore_v4` nem remover pagamentos antigos.

O adaptador universal inclui `ATTACK`, ataques direcionais e pogo na tabela (`src/voidgym/adapters/hollow_knight/adapter.py:27–30`). A máscara em `src/hk_agent/action_masking.py:74–103` filtra ações especiais conforme disponibilidade, mas não desabilita o ataque normal. A confirmação por quadro da máscara e o uso de ataque ficam pendentes da rodada curta em `hk-ai-1`.

## Frequência de ações

### Ginásio — antes

Está em execução a avaliação determinística em 200 salas por nível no split `test`, com checkpoint `E:\VoidGymRuntime\v2\pretrain\pretrain-86d-long-2026-10-29-cuda\universal-009235520`. Como o 86g insere o novo nível 6, a avaliação usa os IDs atuais `0–5, 7, 8`, equivalentes aos níveis anteriores `0–7`. Resultado será salvo em `relatorios/86g_action_frequency_before.json` e resumido no relatório final.

### Hollow Knight — antes

Pendente: capturar uma rodada curta da v2 em `hk-ai-1` quando o operador informar o horário. Não iniciar uma rodada no jogo real sem essa janela. Registrar frequência de `ATTACK` e de ataques direcionais/pogo, contagem de golpes/abates/depósitos e disponibilidade da máscara.
