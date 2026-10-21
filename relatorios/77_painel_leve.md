# Etapa 77 — painel leve e polling controlado

## Resultado

`/api/operator` passou a devolver apenas o resumo da jornada, sem posições e sem SVGs por run. O diário, a lista de campanhas, os relatórios, os candidatos a checkpoint e os registros de lançamento são reutilizados por cache; o cache da jornada invalida pelo mtime/tamanho do diário e do layout. A rota `/api/journey/run?campaign=…&run=…` entrega os detalhes de uma run e limita as posições a 600 pontos amostrados.

O browser faz uma consulta `/api/operator` por vez, com timeout de 10 s, backoff progressivo, pausa em aba oculta e retry manual. A faixa mostra o erro recebido e oferece religar o supervisor quando ele está offline. Consultas de log também não se sobrepõem. O proxy serializa `/api/operator` por cliente. Os gráficos não geram `style=` inline; o seletor apresenta uma opção por campanha com data, rodada e passos e atualiza o rótulo de Continuar com a opção selecionada.

Os rollouts agora carregam cena, progresso máximo observado e eventos confirmados por passo. O banco SIL usa cena, evento e progresso para classificar os trechos persistidos.

## Medições HTTP

| Estado | Porta | Bytes | Tempo |
|---|---:|---:|---:|
| Antes | 8765 | 1.940.694 | 9,666 s |
| Antes | 8766 | 1.940.693 | 9,373 s |
| Depois | 8765 | 56.558 | 196 ms |
| Depois | 8765 | 56.558 | 181 ms |
| Depois | 8765 | 56.557 | 178 ms |
| Depois | 8766 | 56.558 | 176 ms |
| Depois | 8766 | 56.558 | 174 ms |

Redução de payload: aproximadamente **97,1%**. As medições HTTP ficaram abaixo de 300 ms. O endpoint de run retornou 200 posições em 30 ms. O browser carregou a campanha parada em menos de 1 s e mostrou as opções `jogada-1-repeat-1c3b` (R124, 125.952 passos) e `jogada-1` (R137, 123.904 passos), ambas com data.

## Validação

- `pytest -q`: 687 passed, 2 skipped, 8 subtests passed; 3 avisos do SB3.
- Ruff, mypy (55 arquivos), `node --check` e `git diff --check`: aprovados.
- Busca no JavaScript do painel: nenhum atributo `style=` ou `setAttribute('style')` inline.
- Os dois serviços foram reiniciados sem campanhas ativas. Painel em `http://127.0.0.1:8765`, supervisor em `http://127.0.0.1:8766`.
- A validação real de 30 minutos iniciada por Continuar e a observação do painel durante treino estão pendentes de execução nesta etapa.

## Retomada da validação real (20/10/2026)

A campanha `jogada-1-repeat-1c3b` foi retomada pelo endpoint autenticado equivalente ao botão **Continuar**, com um ator e checkpoint R134. O soak de 30 minutos (22:26:46–22:56:46, horário local) terminou com a campanha em execução: ator `COLLECTING`, rodada 151, 5.633 passos coletados e seis relançamentos automáticos registrados. O checkpoint mais recente ao fim da janela era R150 (criado às 22:55:21). Durante a janela, o checkpoint avançou de R146 para R150 e o ator permaneceu com heartbeat atual.

A medição do painel durante a campanha ficou limitada pelo ambiente desta sessão: `127.0.0.1:8765/api/operator` não aceitou conexão (acesso ao socket bloqueado), então não há nova medição HTTP ativa nem confirmação do console CSP nesta validação. A inspeção estática anterior não encontrou atributos `style=` ou `setAttribute('style')` no JavaScript do painel. O soak confirma avanço e persistência de checkpoints; não confirma o alvo de <300 ms sob carga.

Às 23:02:02, depois de 35 minutos, o supervisor encerrou o grupo após o learner reportar `learner heartbeat expirou após 1800s`; o ator terminou como `CANCELLED` com código 0. O último checkpoint persistido foi R154 (23:00:31), com o ator em R155. A sessão não retomou automaticamente a campanha, respeitando a preferência anterior de que o operador inicia o treino.
