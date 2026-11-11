# 76 — Self-imitation learning

## Implementado

- Banco persistente por campanha em `runtime/shared/<campaign>/self-imitation/`, independente da retenção de checkpoints, com amostras NPZ e manifesto escritos atomicamente e limites de 50.000 passos ou 2 GB. A prioridade menor é removida primeiro.
- A seleção atual guarda janelas recorrentes de 32 passos cujo retorno do rollout excede o valor registrado. O learner executa SIL após PPO com estados LSTM, máscaras, coeficientes 0,1/0,01, clipping de gradiente e guardas de entropia/KL.
- O painel tem campo para escolher a campanha de retomada e uma seção para contagem, cenas/eventos, amostras positivas, peso e razão para pular.
- Demonstrações humanas seguem como atualização separada após SIL.

## Validação

- Suíte completa: 686 passed, 2 skipped, 8 subtests passed.
- Testes focados do painel: 88 passed. Ruff em `src tests`, mypy em `src` (55 arquivos), `git diff --check` e `node --check` passaram.
- A campanha exigida `jogada-1-repeat-1c3b` tem checkpoint `round-00000124.zip`; o candidato padrão do painel era outra campanha (`jogada-1`, round 137). Acrescentei seleção explícita, mas, após recarregar, o painel ficou em “Conectando ao painel…”, sem preencher o seletor. A API GET `/api/operator` responde no terminal em aproximadamente 14 s, enquanto a página não conclui a atualização.
- Não iniciei uma campanha diferente. O painel indica 0 instâncias ativas. Não foi feito o soak de 30 minutos nem retomada real, portanto não há medida posterior para comparação.
- Antes da SIL, o painel mostrava 1,2–1,3 passos/s, 5 ajustes/h e máximo de 131,4 u em Tutorial_01.

## Limitação dos dados de seleção

O formato atual de rollout compartilhado não inclui cena nem tipo de evento por transição. Por isso o banco marca amostras como `scene=unknown`, `event=high_return` e seleciona apenas retorno acima do valor registrado. Recordes de progresso, eventos raros, bônus de raridade e diversidade real por cena/evento ainda não estão implementados.
