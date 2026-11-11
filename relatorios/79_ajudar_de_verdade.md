# Etapa 79 — Ajuda ao vivo

## Entrega

O painel mostra os pedidos logo abaixo do banner, com contador amarelo, miniatura, local, travamentos, tempo aberto e ações **Ajudar agora**, **Agora não (30 min)**, **Ver no mapa**, **Pronto** e **Descartar**. O histórico de pedidos e ajudas agora fica em uma seção própria. Quando `/api/operator` demora, o painel tenta a rota leve `/api/help-requests` usando a campanha lembrada localmente. A bandeja existente recebe as mudanças pela lista de pedidos.

**Ajudar agora** muda apenas o slot correspondente para `HELPING`. O bridge envia `RELEASE_ALL`, restaura a janela daquele processo e troca de volta para o teclado/controle físico. Ignora leases do agente enquanto esse modo está ativo e envia botões/duração no `FRAME`. O ator continua com heartbeat `HELPING`; o learner pausa esse slot, remove-o do quórum de rollout e segue com os demais. Se todos estiverem ajudando, o learner permanece vivo em `WAITING_HELP`.

A tarefa grava o PNG da observação recebida pela política, o histórico de ações, conhecimento do jogador e máscara de ações, na mesma cadência. A cobertura e o tempo aparecem no painel; cobertura abaixo de 70% recebe aviso. A gravação termina ao mudar de cena ou ao afastar-se mais de duas células do ponto inicial. **Pronto** salva primeiro, devolve o controle à política e marca `resolved_with_help`; o learner existente carrega o arquivo pelo contexto e registra `demonstration_training` com a perda. Uma demonstração sem ações utilizáveis é descartada. **Descartar** não cria artefato. A ajuda expira em dez minutos e devolve o controle à IA; o bridge também tem o mesmo limite local.

O bridge oferece `HK_AGENT_HUMAN_INPUT_TEST=1` para alternar os botões roteirizados `RIGHT` e `RIGHT+JUMP`; o `FRAME` e o manifesto marcam a fonte como teste. O padrão é entrada física.

## Segurança e operação

O processo de ajuda controla somente sua própria janela e a gravação fica em `shared/demonstrations/<contexto>`. Não alterei saves pessoais nem baseline. A DLL foi implantada somente nas cópias isoladas `hk-ai-1` e `hk-ai-2`; os DLLs anteriores estão preservados em `C:\Users\thayr\Games\HollowKnight-AI\bridge-backups\79-20261020-233513`. Não abri jogo nem alterei o ledger da campanha principal. Na conferência somente leitura, o pedido citado no prompt já estava `resolved_solo` (a IA passou do local); os outros dois pedidos do ledger também estavam resolvidos. Portanto não havia pedido aberto para manter no cartão. A instrução anterior foi deixar o início do treino com o operador; por isso, a validação E2E foi sintética, sem janela de jogo.

Instrução ao operador: **clique Ajudar agora → a janela da instância vem para frente → jogue até passar → clique Pronto**. Use **Descartar** para devolver o controle sem guardar a demonstração. A prática curta de retorno ao ponto foi omitida: não encontrei um snapshot de retorno associado a este pedido.

## Validação

- `pytest`: 693 passaram, 2 ignorados, 8 subtestes passaram; 3 avisos preexistentes do Stable Baselines.
- `ruff check src tests`, `mypy src`, `node --check` e `git diff --check`: passaram.
- `dotnet build -c Release` do bridge: passou, 0 avisos e 0 erros.
- O teste sintético percorre ação roteirizada marcada como teste, gravação, carregamento no formato do learner e confirmação da perda de imitação. O teste do quórum confirma que `HELPING` não vira straggler.

## Limites

A demonstração real permanece intocada conforme pedido. Não rodei campanha com janela de jogo; a passagem da fonte roteirizada pelo processo do jogo ainda precisa de observação em campanha de teste iniciada pelo operador. A UI também não mostra a perda real do learner nesta campanha parada; essa confirmação ocorre no manifesto após o próximo rollout treinável.
