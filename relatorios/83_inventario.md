# Inventário da v1 para a v2

## Observação

`src/hk_agent/gym_env.py:220-230` cria o `Dict` da política no construtor de `HollowKnightEnv`.
Ele contém `pixels`, shape `(frame_stack,H,W,3)`, `uint8`, e `action_history`, `int64`, com
comprimento `frame_stack - 1`. O ambiente guarda quadros RGB e o histórico das últimas ações.
O conhecimento do jogador e o bestiário são fontes auxiliares/telemetria, não chaves da
observação do ambiente visual principal.

## Ações

`src/hk_agent/temporal_actions.py:42-88` define 35 ações de combate e `:90-99` define 8 de
interação/menu. A lista ordenada é: `NOOP`, `LEFT`, `RIGHT`, `JUMP_TAP`, `JUMP_HOLD`,
`LEFT+JUMP_TAP`, `LEFT+JUMP_HOLD`, `RIGHT+JUMP_TAP`, `RIGHT+JUMP_HOLD`, `JUMP_MAX`,
`LEFT+JUMP_MAX`, `RIGHT+JUMP_MAX`, `ATTACK`, `LEFT+ATTACK`, `RIGHT+ATTACK`, `UP+ATTACK`,
`DOWN+ATTACK`, `JUMP+ATTACK`, `LEFT+JUMP+ATTACK`, `RIGHT+JUMP+ATTACK`, `CAST`, `UP+CAST`,
`DOWN+CAST`, `FOCUS_HOLD`, `NAIL_ART_CHARGE`, `LEFT+NAIL_ART_CHARGE`, `RIGHT+NAIL_ART_CHARGE`,
`UP+NAIL_ART_CHARGE`, `DOWN+NAIL_ART_CHARGE`, `DASH`, `LEFT+DASH`, `RIGHT+DASH`, `DOWN+DASH`,
`SUPER_DASH_HOLD`, `DREAM_NAIL`, `INTERACT`, `MENU_UP`, `MENU_DOWN`, `MENU_LEFT`, `MENU_RIGHT`,
`MENU_SUBMIT`, `MENU_CANCEL`, `OPEN_CHARMS`.
`TemporalActionTemplate` guarda nome, botões e duração (`base`, `tap`, `hold`, `max_hold`,
`focus`, `charge`) em `temporal_actions.py:14-34`; `action_contract()` está em `:142`. A máscara
é calculada por `src/hk_agent/action_masking.py:74`.

## Fronteira do jogo

`src/hk_agent/gym_env.py` implementa o contrato síncrono `EnvironmentBackend`, lease, aplicação,
captura, release e telemetria por `PixelFrame` (`:84-110`, `:304+`, `_observation():575`,
`_info():584`). `menu_bootstrap.py` e `menu_session.py` cuidam do
menu inicial; `supervisor.py`, `supervisor_backend.py` e `process_owner.py` mantêm processos,
recuperação e isolamento; troca de cena é reconhecida em `gym_env.py` e `scene_change_task.py`.

## Recompensa

O ambiente visual atual retorna recompensa zero. Tarefas de exploração e combate estão em
`exploration_task.py`, `combat_reward.py` e `training_task.py`; usam progresso de sala, posição,
HP, morte, eventos de inimigos, geo, sombra e mudanças de cena.

## Política

`recurrent_sb3.py` contém o extrator visual recorrente e cabeças de ação/valor; `first_training.py`
configura treino inicial; `shared_learner.py` coordena PPO, checkpoints e avaliações.

## Universal e específico do HK

- Universal: pixels, histórico de ações, interfaces Gymnasium, sequência recorrente e objetivos
  simples de movimento; esses conceitos se aplicam a jogos 2D.
- Específico do HK: bridge/Modding API, cenas, saves, HP/alma, geo, sombra, banco, bestiário,
  menu bootstrap e nomes de ações; dependem de Hollow Knight e ficam no adaptador.

Inventário feito a partir do código local; números de linha podem mudar com as próximas etapas.
