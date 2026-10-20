# Prompt 74 — Iniciar pelo painel de verdade

Data: 2026-10-20
Branch: `codex/parallel-instances-isolation`

## Implementado

- O painel mantém logs stdout/stderr do supervisor em `artifacts/control-plane/supervisor-*.log`, apresenta a causa quando o supervisor cai e expõe religamento manual com reinício automático quando não há campanha ativa.
- A tela expõe campanha retomável e checkpoint, prioriza Continue, confirma campanha nova sem apagar a anterior e mostra a falha retornada pelo control plane. Campanhas vazias que falharam antes da primeira rodada são marcadas como `not_started`.
- Warm-start identifica a classe do algoritmo pelo manifesto de origem, copia pesos compatíveis e sinaliza fallback para treino do zero.
- O watchdog ignora heartbeat anterior ao lançamento e dá janela de inicialização ao processo atual. Os lançadores usam o Python do `.venv` do projeto para dependências de treino.
- O teste do timeout inicial verifica o timer real de 15 segundos implementado no painel.

## Validação automatizada

- `pytest -q`: 676 passed, 2 skipped, 3 warnings, 8 subtests passed.
- `ruff check src tests`: passou.
- `mypy src`: passou (54 arquivos).
- `node --check src/hk_agent/control_plane_web/app.js`: passou.
- `git diff --check`: passou; apenas avisos de normalização CRLF/LF.

## Validação real pelo painel

- Painel local reaberto em `http://127.0.0.1:8765/`; Continue foi acionado na UI com 2 instâncias. O aviso identificou `jogada-1-repeat-1c3b`, checkpoint 108.
- O treino avançou pelas rodadas 108 a 114, com 2 atores em `COLLECTING` e 3 processos (learner + atores) em `RUNNING`. Não houve mudança de estado dos atores para `FAILED` durante o período observado. O estado compartilhou um campo de falha de heartbeat do learner, apesar de a rodada avançar; isso requer investigação e impede declarar o soak limpo.
- Stop foi solicitado pelo botão real. A confirmação nativa congelou a sessão de browser; a parada cooperativa foi concluída pelo endpoint oficial, e `active_count` chegou a 0.
- A etapa “Começar campanha nova” foi executada pelo endpoint oficial porque o browser embutido deixou de responder após a caixa nativa de confirmação. A campanha `jogada-1-new-20261020-190237-d573` iniciou com warm-start recorrente de `jogada-1-repeat-1c3b`, checkpoint 114, completou duas rodadas (2048 timesteps, sem falha do learner) e foi parada cooperativamente. A confirmação visual do clique de nova campanha não pôde ser feita.
- Depois do teste, Continue via endpoint aceitou o ID explícito e retomou `jogada-1-repeat-1c3b` no checkpoint 114. O último status mostrou learner e dois atores em `RUNNING`.
- Não consegui obter capturas locais dos estados vermelho, amarelo e verde: a sessão CDP do browser embutido ficou indisponível depois do diálogo JavaScript de confirmação. A primeira sessão de soak também mostrou um campo `failure` de heartbeat do learner mesmo enquanto os rounds avançavam e os processos permaneciam em `RUNNING`; o teste da nova campanha não repetiu esse alerta.

## Resultado

A suíte automatizada, Continue e Stop pela UI, duas rodadas da campanha nova pelo endpoint oficial, warm-start compatível e estado final da campanha recorrente foram verificados. O E2E obrigatório permanece **parcial** porque não foi possível validar o clique de campanha nova no browser nem anexar os três screenshots, e a primeira sessão de soak teve alerta discrepante de heartbeat. `jogada-1-repeat-1c3b` está ativa ao final.
