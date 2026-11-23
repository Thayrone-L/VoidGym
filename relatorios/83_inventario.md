# Prompt 83 — inventário da v1

Data: 2026-10-21. Esta etapa só lê o código v1 e cria este inventário; nenhum arquivo em `src/hk_agent/` foi alterado.

## 1. Observação

O ambiente base `HollowKnightEnv` cria o `Dict` em `src/hk_agent/gym_env.py:190-232`. Na configuração usada pelo launcher (`tools/run_first_training.py:105,113-115,135-136`), a observação antes do wrapper da política contém:

| Chave | Shape | dtype | Conteúdo |
| --- | --- | --- | --- |
| `pixels` | `(4, 338, 600, 3)` | `uint8` | Pilha RGB, canais no último eixo. Os tamanhos de entrada são configuráveis. |
| `action_history` | `(3,)` | `int64` | Três ações aplicadas anteriormente; `temporal_observation_space` altera os limites para a quantidade de ações da tarefa (`src/hk_agent/temporal_actions.py:102-118`). |

Em `explore_v4`, `ExplorationTask` acrescenta `player_knowledge` (`src/hk_agent/exploration_task.py:461-469,1642-1650`): vetor `(1555,)`, `float32`, valores limitados a `[0,1]` (`src/hk_agent/player_knowledge.py:486-563`; tamanho confirmado por `PlayerKnowledge.space()`). O vetor reúne itens/habilidades/charmes, mapa visitado e saídas em grade 16×9, lugares e objetivos da história, dicas, preços de loja, bestiário e charmes equipados. O bloco de bestiário tem 6 slots × 13 valores (`src/hk_agent/bestiary.py:9-11,170-176`). Versões anteriores a v4 não adicionam `player_knowledge`.

**Diferença encontrada:** o launcher envolve a tarefa em `ChannelsFirstObservation` (`tools/run_first_training.py:659-670`). O wrapper declara e devolve somente `pixels` e `action_history` (`src/hk_agent/training_task.py:475-499`), descartando `player_knowledge`. Assim, o contrato da tarefa declara três chaves em v4, mas a observação entregue à política atual tem apenas duas: `pixels` com shape `(12, 90, 160)`, `uint8`, e `action_history` `(3,)`, `int64`. O redimensionamento usa amostragem por índices `linspace`, sem letterbox (`training_task.py:470-498`). Para preservar a v1 e checkpoints, a fundação v2 deve registrar essa diferença sem alterar o wrapper nem afirmar que a política atual recebe o vetor de conhecimento.

## 2. Ações

`explore_v4` usa `SCENE_CHANGE_INTERACTION_ACTION_TEMPLATES`, a concatenação das 35 ações de movimento/combate e 8 ações de interação/menu (`src/hk_agent/scene_change_task.py:34-43`; seleção em `src/hk_agent/exploration_task.py:451-459`). `resolve_lease_ms` define os tempos em `src/hk_agent/temporal_actions.py:27-39`; com lease-base de 80 ms, os tempos efetivos são:

| Índice | Nome | Botões | Tipo | Lease (ms) |
|---:|---|---|---|---:|
| 0 | `NOOP` | — | base | 80 |
| 1 | `LEFT` | LEFT | base | 80 |
| 2 | `RIGHT` | RIGHT | base | 80 |
| 3 | `JUMP_TAP` | JUMP | tap | 50 |
| 4 | `JUMP_HOLD` | JUMP | hold | 160 |
| 5 | `LEFT+JUMP_TAP` | LEFT, JUMP | tap | 50 |
| 6 | `LEFT+JUMP_HOLD` | LEFT, JUMP | hold | 160 |
| 7 | `RIGHT+JUMP_TAP` | RIGHT, JUMP | tap | 50 |
| 8 | `RIGHT+JUMP_HOLD` | RIGHT, JUMP | hold | 160 |
| 9 | `JUMP_MAX` | JUMP | max_hold | 450 |
| 10 | `LEFT+JUMP_MAX` | LEFT, JUMP | max_hold | 450 |
| 11 | `RIGHT+JUMP_MAX` | RIGHT, JUMP | max_hold | 450 |
| 12 | `ATTACK` | ATTACK | tap | 50 |
| 13 | `LEFT+ATTACK` | LEFT, ATTACK | tap | 50 |
| 14 | `RIGHT+ATTACK` | RIGHT, ATTACK | tap | 50 |
| 15 | `UP+ATTACK` | UP, ATTACK | tap | 50 |
| 16 | `DOWN+ATTACK` | DOWN, ATTACK | tap | 50 |
| 17 | `JUMP+ATTACK` | JUMP, ATTACK | base | 80 |
| 18 | `LEFT+JUMP+ATTACK` | LEFT, JUMP, ATTACK | base | 80 |
| 19 | `RIGHT+JUMP+ATTACK` | RIGHT, JUMP, ATTACK | base | 80 |
| 20 | `CAST` | CAST | tap | 50 |
| 21 | `UP+CAST` | UP, CAST | tap | 50 |
| 22 | `DOWN+CAST` | DOWN, CAST | tap | 50 |
| 23 | `FOCUS_HOLD` | CAST | focus | 1500 |
| 24 | `NAIL_ART_CHARGE` | ATTACK | charge | 1500 |
| 25 | `LEFT+NAIL_ART_CHARGE` | LEFT, ATTACK | charge | 1500 |
| 26 | `RIGHT+NAIL_ART_CHARGE` | RIGHT, ATTACK | charge | 1500 |
| 27 | `UP+NAIL_ART_CHARGE` | UP, ATTACK | charge | 1500 |
| 28 | `DOWN+NAIL_ART_CHARGE` | DOWN, ATTACK | charge | 1500 |
| 29 | `DASH` | DASH | tap | 50 |
| 30 | `LEFT+DASH` | LEFT, DASH | tap | 50 |
| 31 | `RIGHT+DASH` | RIGHT, DASH | tap | 50 |
| 32 | `DOWN+DASH` | DOWN, DASH | tap | 50 |
| 33 | `SUPER_DASH_HOLD` | SUPER_DASH | charge | 1500 |
| 34 | `DREAM_NAIL` | DREAM_NAIL | hold | 160 |
| 35 | `INTERACT` | UP | tap | 50 |
| 36 | `MENU_UP` | UP | tap | 50 |
| 37 | `MENU_DOWN` | DOWN | tap | 50 |
| 38 | `MENU_LEFT` | LEFT | tap | 50 |
| 39 | `MENU_RIGHT` | RIGHT | tap | 50 |
| 40 | `MENU_SUBMIT` | MENU_SUBMIT | tap | 50 |
| 41 | `MENU_CANCEL` | MENU_CANCEL | tap | 50 |
| 42 | `OPEN_CHARMS` | PAUSE | tap | 50 |

The legal mask is built in `src/hk_agent/action_masking.py:74-111` and applied in `exploration_task.py:821-830`. It uses bridge `action_availability` for abilities, soul, menus, charms, and bench state; missing availability disables selected actions conservatively.

## 3. Jogo e bridge

- Task action name → button tuple and resolved lease: `exploration_task.py:821-830`, `temporal_actions.py:27-39,142-154`.
- Environment assigns action ID and calls `backend.step`, validates the acknowledged action and captured frame, advances frame/action history, and turns operational failures into an invalid truncation: `gym_env.py:295-420`.
- `SupervisorBackend.step` sends `APPLY_ACTION`, waits for lease plus settle time, always sends `RELEASE_ALL`, then sends `CAPTURE_ONCE`; the frame must match the action ID (`supervisor_backend.py:452-508,552-579`).
- IPC command correlation, timeout, and `FRAME` decoding/validation live in `supervisor.py:212-295,300-525`. Frame telemetry includes pixels and game-specific fields such as scene transition, death, HP/geo deltas, shade state/distance, enemy/milestone events, and action availability.
- Reset starts an episode, captures its first frame, and fills the initial four-frame stack by repeating that frame (`gym_env.py:251-293`). `release_all` runs at reset/close, on termination, and on failures (`gym_env.py:259-283,350-358,416-431`).
- Menu bootstrap is a separate isolated sequence: `src/hk_agent/menu_bootstrap.py:271-299` captures screens and acknowledges `RELEASE_ALL`; process/session startup and cleanup are in `src/hk_agent/isolated_session.py`.
- Scene-transition frames are rejected by `gym_env.py:483-495`; the next valid frame is required before training continues. Environment/backend exceptions trigger best-effort release and truncation (`gym_env.py:350-358`).
- Stuck recovery at the task level is a PPO-slice truncation after elapsed lease time without progress, not a bench reload (`exploration_task.py:1481-1505`). Process ownership and actor relaunch/straggler handling are coordinated by `process_owner.py`, `episode_worker.py`, and the shared learner/controller.

## 4. Recompensa `explore_v4`

Reward shaping is in `src/hk_agent/exploration_task.py`, primarily `step` (`769+`, component calculations `~1000-1481`). It combines visual novelty and scene changes with bridge-confirmed enemy kills, HP loss/restoration, Geo gained/spent, Shade recovery/hits/distance, death/run failure, and v4 milestone/tutorial/challenge events. Reward components are budgeted and terminal transitions suppress positive awards. The v4 task contract lists the reward-only signals and explicitly separates them from the policy boundary (`exploration_task.py:1935-1993`). Scene/layout positions and event identity are Hollow Knight signals; the policy claim is pixel-based.

## 5. Política

- Image feature extraction and channel conversion: `src/hk_agent/training_task.py:457-499`; current SB3 image is `(12,90,160)` and action history is a separate Dict feature.
- Recurrent policy: `src/hk_agent/recurrent_sb3.py:99-214` extends sb3-contrib's multi-input recurrent actor-critic. `CombinedExtractor` supplies image/vector features; SB3-contrib supplies actor and critic LSTMs; inherited policy heads include action logits and value estimate. `MaskableRecurrentPPO` owns the recurrent masked rollout/update path (`235+`).
- Single-process PPO construction/resume/checkpoint: `src/hk_agent/first_training.py:211+`, model creation `~344-376`.
- Shared policy learner, recurrent rollout import and SIL updates: `src/hk_agent/shared_learner.py:1943+`, model selection `~2013-2034`, SIL integration `~2221+`. Actor/learner settings and task contract carry the action and observation shapes.

## 6. Limite entre universal e específico

### Reaproveitável para jogos 2D

- Pilha de quadros, imagem RGB, histórico recente de ações e wrappers Gymnasium: conceitos independentes do título.
- Ações com direção, pulo, ataque, especial e duração: vocabulário transferível se o jogo tiver equivalentes.
- Loop síncrono `reset/step`, máscara booleana, função de recompensa injetável e política recorrente: interfaces úteis para outros jogos, depois de separar dependências do bridge.
- Medidas de novelty por imagem e avaliação por passos: podem ser reutilizadas com parâmetros próprios do jogo.

### Específico de Hollow Knight

- Botões `CAST`, `DREAM_NAIL`, `SUPER_DASH`, menu de charms, bancos, soul e custos/níveis de feitiços.
- Bridge Unity/BepInEx, comandos `APPLY_ACTION`/`CAPTURE_ONCE`/`RELEASE_ALL`, formato `FRAME`, saves e isolamento de instâncias.
- Cena/nome de sala, HP, soul, Geo, Shade, inimigos, Grubs, chefes, itens, charms, estações, história e milestones.
- `PlayerKnowledge`, bestiário, flags de progressão, objetivos e semântica de recompensa `explore_v4`.
- Bootstrap de menus, leitura de saves e recuperação de Shade/death: regras operacionais de HK que não cabem no núcleo universal.

## Resultado da Etapa 0

O inventário confirma que a v1 tem bons conceitos transferíveis, mas também encontrou o descarte de `player_knowledge` no wrapper da política. A opção segura é deixar a v1 e seus checkpoints intactos e registrar essa discrepância como parte do contrato que o adaptador deve espelhar explicitamente.
