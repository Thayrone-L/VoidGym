# Etapa 75 — Avaliações curtas, alternadas e visíveis

## Implementação

- As provas avaliam um único ator por fronteira e alternam os slots. O outro ator segue coletando com a política publicada.
- Cada prova usa política determinística e estocástica. Ela encerra após 30 s ou 300 passos sem progresso e nunca ultrapassa 1.000 passos.
- O registro guarda medida, progresso máximo, tempo e passos até travar, motivo do fim e se a prova terminou cedo. Provas não entram no diário de runs nem nos rollouts de treino.
- O painel identifica a prova em andamento, o progresso até 1.000 passos e exibe as duas medidas com seus resultados. A revisão incluiu um bug descoberto no ensaio: a chave do diário deduplicava por rodada/ator e descartava a segunda medida. A chave agora inclui a medida.
- A prática de travessia tem limite de dez tentativas por sessão, registra o motivo de conclusão, mostra `Praticando trecho X (tentativa k/M)` e retorna à jogada normal.

## Validação

- Suíte completa: **681 passaram, 2 ignorados, 8 subtests passaram**. Três avisos do Stable-Baselines3 sobre cópia de tensores.
- `ruff check src tests`: passou.
- `mypy src`: passou, 54 arquivos.
- `node --check src/hk_agent/control_plane_web/app.js`: passou.
- `git diff --check`: passou; Git exibiu avisos informativos de conversão CRLF/LF em dois arquivos.
- Painel e jogo, campanha temporária `probe75final-20261020`, dois atores, intervalo temporário 1 e rollout normal de 512:
  - Round 1: slot 0 avaliado; o slot 1 continuou em `COLLECTING`. A prova determinística terminou por travamento em 222 passos / 30,11 s; a estocástica, em 217 / 30,09 s.
  - O learner avançou para round 2. O slot 1 foi avaliado enquanto o slot 0 coletava. A medida determinística terminou por travamento em 225 / 30,11 s; a estocástica atingiu o limite em 1.000 passos.
  - O learner avançou para round 3 e a alternância voltou ao slot 0. O painel mostrou as duas medidas, progresso, duração e passos.
  - Encerrei a campanha temporária por parada segura após a validação. Estado final: `active_count=0`. A especificação da campanha original permaneceu em `evaluation_interval=20`.

A campanha temporária foi criada com checkpoint warm-start da `jogada-1-repeat-1c3b`; os saves e dados de runtime da validação permanecem no runtime local.
