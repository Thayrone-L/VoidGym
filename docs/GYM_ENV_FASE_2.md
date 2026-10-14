# Ambiente Gymnasium da Fase 2

`hk_agent.gym_env.HollowKnightEnv` implementa a fronteira de aprendizagem sem
conter uma política, rota ou estratégia de Hollow Knight. Nesta fase toda
recompensa é `0.0`.

## Espaços

A observação é um `gymnasium.spaces.Dict`:

- `pixels`: `uint8 [K,H,W,3]`, com frames RGB empilhados em ordem temporal;
- `action_history`: os índices das `K-1` ações anteriores.

A ação é `Discrete(1024)`: um bitmask dos dez botões brutos, todos com duração
fixa. O bit `n` corresponde à posição `n` nesta ordem:

| Bit | Entrada |
|---:|---|
| 0–3 | `LEFT`, `RIGHT`, `UP`, `DOWN` |
| 4–9 | `JUMP`, `ATTACK`, `DASH`, `CAST`, `SUPER_DASH`, `DREAM_NAIL` |

O índice zero é `NOOP`. Combinações são simultâneas e mecânicas, não macros.
Se esquerda+direita ou cima+baixo forem pedidos juntos, ambos os botões do eixo
são removidos. A ação original e a ação normalizada são registradas. Não há
controle de menu ou conhecimento de habilidades disponíveis.

## Backend

O `EnvironmentBackend` é injetável e síncrono. Ele associa cada reset a um
`episode_id` novo e cada passo a um `action_id`, devolvendo a ação efetivamente
aplicada e um frame correlacionado. O ambiente rejeita frames duplicados, fora
de ordem, com timestamp regressivo, durante transição, de outro episódio ou sem
o `last_applied_action_id` esperado. A rejeição trunca o episódio, executa
`RELEASE_ALL` e não entra no log de treino.

O backend real é `hk_agent.supervisor_backend.SupervisorEnvironmentBackend`.
Ele mantém o `SupervisorServer` em uma thread com loop `asyncio` dedicado e
expõe a interface síncrona exigida pelo Gymnasium. `BEGIN_EPISODE` associa um
UUID novo na ponte; `APPLY_ACTION` exige esse UUID e o `FRAME` devolve tanto o
`episode_id` quanto o último `action_id` aplicado. O PNG RGB8 é validado,
descomprimido com limite e convertido em `numpy.uint8 [H,W,3]` antes de entrar
no ambiente.

A ponte mantém um latch de morte reiniciado por `BEGIN_EPISODE`. Quando o hook
`AfterPlayerDeadHook` dispara, o próximo `FRAME` traz `player_dead=true`; o
backend converte isso em `terminated=True` com razão `player_death`. Isso encerra
somente o episódio: Gym/PPO solicita outro `reset()` e a campanha continua. Já
falhas de protocolo, captura ou isolamento continuam sendo falhas operacionais.

O backend não inicia Hollow Knight por conta própria e não conhece menu, mapa,
rota ou estratégia. Para resetar processos, injete `reset_process(endpoint,
episode_id)`: esse callback deve apenas encerrar/restaurar/iniciar a instância
e o save isolado, deixando a nova ponte conectar ao endpoint informado. Sem
callback, somente o primeiro episódio de uma ponte já conectada é aceito; um
novo `reset()` é rejeitado em vez de inventar uma sequência de menu. Um
`shutdown_process` opcional permite ao proprietário encerrar o processo no
fechamento, sem impedir o cleanup do servidor caso o callback falhe. Ambos os
callbacks são executados em threads daemon e limitados por
`process_timeout_seconds`; timeout invalida o episódio e nunca bloqueia o
encerramento do backend. Um timeout de `reset_process` também fecha
definitivamente o listener e o backend, impedindo que uma ponte iniciada tarde
seja aceita como uma geração posterior. Como Python não pode cancelar à força
uma função arbitrária, esses callbacks fazem parte da infraestrutura confiável:
devem ser idempotentes, respeitar seus próprios prazos e não adiar efeitos
externos para depois do retorno.

Cada `step()` envia uma única ação bruta, aguarda o lease, executa
`RELEASE_ALL` e só então pede o frame correlacionado. Operações públicas são
serializadas; todos os comandos possuem timeout e `close()` é idempotente.

## Persistência e encerramento

Cada transição aceita é anexada e sincronizada em
`instances/<instance>/episodes/<episode>/transitions.jsonl`. Falhas e
transições rejeitadas vão para `instances/<instance>/operations.jsonl` e nunca
para o rollout. Uma falha de persistência também trunca o episódio antes de o
estado observável avançar.

`close()` é idempotente e sempre tenta fechar o backend mesmo se
`RELEASE_ALL` falhar. `terminated` é reservado para morte/sucesso real;
limite de passos e falhas operacionais usam `truncated`.
