# 76 — Self-imitation learning

## Implementado

- Banco por campanha em `runtime/shared/<campaign>/self-imitation/`, separado dos checkpoints, com gravação atômica NPZ/manifest, persistência e limites de 50.000 passos ou 2 GB; amostras de menor prioridade são removidas primeiro.
- Seleção atual guarda janelas recorrentes de 32 passos cujo retorno do rollout excede o valor registrado. A atualização SIL roda depois do PPO com os estados LSTM salvos, máscaras de ação, coeficientes 0,1/0,01, clipping de gradiente e guardas de entropia/KL.
- O learner publica métricas da última atualização; o endpoint e o painel exibem contagem, eventos/cenas, amostras positivas, peso e razão para pular.
- As demonstrações humanas continuam sendo aplicadas separadamente após SIL.

## Validação

- Suíte completa: 686 passed, 2 skipped, 8 subtests passed, incluindo atualização recorrente mascarada SIL.
- Ruff: aprovado. Mypy em `src`: aprovado (55 arquivos). `git diff --check`: aprovado. `node --check` no painel: aprovado.
- Não houve soak de 30 minutos nem retomada de jogo nesta execução. O painel está parado, mas seu botão de continuação oferece `jogada-1 v4 · rodada 137`; o prompt pede especificamente `jogada-1-repeat-1c3b`. Não iniciei uma campanha diferente por engano.
- Antes da SIL, o painel mostrava 1,3 passos/s, 5,0 ajustes/h e progresso máximo 131,4 u em Tutorial_01. A comparação posterior e as avaliações determinística/estocástica dependem do soak ainda não executado.

## Limitações de seleção

O formato de rollout compartilhado não contém cena nem tipo de evento por transição. Por isso as amostras são identificadas como `scene=unknown`, `event=high_return`; eventos raros, recordes de progresso, bônus de raridade e diversidade real por cena/evento ainda não são selecionados. A implementação não afirma ter validado esses critérios.

