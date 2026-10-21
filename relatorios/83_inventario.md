# Inventário v1 para a v2 (prompt 83)

Inventário feito por leitura do código; o jogo, a DLL e os checkpoints não existem nesta máquina.

## 1. Observação

- `src/hk_agent/gym_env.py:181-230`: `HollowKnightEnv` cria `pixels` como `(K, H, W, 3)`, `uint8`, e `action_history` como `(K-1,)`, `int64` (MultiDiscrete). K/H/W são parâmetros do ambiente; o lançador registra dimensões em `tools/run_scene_change_probe.py`.
- `src/hk_agent/temporal_actions.py:102-120`: `temporal_observation_space` substitui o vocabulário do histórico pelo tamanho da política.
- `src/hk_agent/exploration_task.py:451-469`: a política usa 35 ações no modo scene-change e 43 no modo `explore_v4`; somente `explore_v4` inclui `player_knowledge`.
- `player_knowledge`: Box `float32`, shape `(1555,)`, em `src/hk_agent/player_knowledge.py` (itens, exploração própria, pontos de interesse, tutoriais e bestiário pessoal).
- A observação real não tem um campo separado para o bestiário: ele faz parte do vetor `player_knowledge`.
- No contrato proposto da v2, pixels passam a `pixels` `(12,108,192)` `uint8` (4 quadros RGB em canais) e `action_history` segue separado.

## 2. Ações e máscara

`src/hk_agent/temporal_actions.py:42-99` define as ações; `src/hk_agent/scene_change_task.py:34-43` escolhe 35 ou 43; `src/hk_agent/action_masking.py:74-110` calcula disponibilidade. A tabela de 43 ações `explore_v4`, com o lease para base 80 ms:

| Índice | Nome | Botões | Duração | Lease ms |
|---:|---|---|---|---:|
| 0 | `NOOP` | — | base | 80 |
| 1 | `LEFT` | LEFT | base | 80 |
| 2 | `RIGHT` | RIGHT | base | 80 |
| 3 | `JUMP_TAP` | JUMP | tap | 50 |
| 4 | `JUMP_HOLD` | JUMP | hold | 160 |
| 5 | `LEFT+JUMP_TAP` | LEFT+JUMP | tap | 50 |
| 6 | `LEFT+JUMP_HOLD` | LEFT+JUMP | hold | 160 |
| 7 | `RIGHT+JUMP_TAP` | RIGHT+JUMP | tap | 50 |
| 8 | `RIGHT+JUMP_HOLD` | RIGHT+JUMP | hold | 160 |
| 9 | `JUMP_MAX` | JUMP | max_hold | 450 |
| 10 | `LEFT+JUMP_MAX` | LEFT+JUMP | max_hold | 450 |
| 11 | `RIGHT+JUMP_MAX` | RIGHT+JUMP | max_hold | 450 |
| 12 | `ATTACK` | ATTACK | tap | 50 |
| 13 | `LEFT+ATTACK` | LEFT+ATTACK | tap | 50 |
| 14 | `RIGHT+ATTACK` | RIGHT+ATTACK | tap | 50 |
| 15 | `UP+ATTACK` | UP+ATTACK | tap | 50 |
| 16 | `DOWN+ATTACK` | DOWN+ATTACK | tap | 50 |
| 17 | `JUMP+ATTACK` | JUMP+ATTACK | base | 80 |
| 18 | `LEFT+JUMP+ATTACK` | LEFT+JUMP+ATTACK | base | 80 |
| 19 | `RIGHT+JUMP+ATTACK` | RIGHT+JUMP+ATTACK | base | 80 |
| 20 | `CAST` | CAST | tap | 50 |
| 21 | `UP+CAST` | UP+CAST | tap | 50 |
| 22 | `DOWN+CAST` | DOWN+CAST | tap | 50 |
| 23 | `FOCUS_HOLD` | CAST | focus | 1500 |
| 24 | `NAIL_ART_CHARGE` | ATTACK | charge | 1500 |
| 25 | `LEFT+NAIL_ART_CHARGE` | LEFT+ATTACK | charge | 1500 |
| 26 | `RIGHT+NAIL_ART_CHARGE` | RIGHT+ATTACK | charge | 1500 |
| 27 | `UP+NAIL_ART_CHARGE` | UP+ATTACK | charge | 1500 |
| 28 | `DOWN+NAIL_ART_CHARGE` | DOWN+ATTACK | charge | 1500 |
| 29 | `DASH` | DASH | tap | 50 |
| 30 | `LEFT+DASH` | LEFT+DASH | tap | 50 |
| 31 | `RIGHT+DASH` | RIGHT+DASH | tap | 50 |
| 32 | `DOWN+DASH` | DOWN+DASH | tap | 50 |
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

A máscara valida ações conforme `ActionAvailability` (habilidades, interface atual e botões disponíveis); a política não recebe os sinais brutos usados para calculá-la.

## 3. Comunicação com o jogo

- `src/hk_agent/gym_env.py:299-389`: valida ação/lease, chama `backend.step`, valida correlação e aplica captura/frames.
- `src/hk_agent/supervisor_backend.py:300-370, 550-710`: lease/apply/capture/release, validação de ACK e FRAME.
- `src/hk_agent/supervisor.py:293-494`: decodifica e valida FRAME, dimensões, imagem e telemetria.
- `src/hk_agent/control_plane.py` e `control_plane_server.py`: supervisor do operador, bootstrap/menu e ciclo de campanha; `src/hk_agent/menu_bootstrap.py` realiza a recuperação de menu.
- `src/hk_agent/process_owner.py` e `isolated_session.py`: início/fim isolados, saves experimentais, restauração em falha. `scene_change_task.py` e `exploration_task.py` encerram fatias, detectam progresso e travadas.
- A troca de cena chega nos metadados do FRAME e é consumida pelo ambiente/tarefa. Não houve execução do jogo nesta máquina.

## 4. Recompensa

- `src/hk_agent/exploration_task.py:769+` e `:1240-1640`: `explore_v4` agrega novidade por célula, recorde de progresso, dano/morte, tempo, sala, coleta, banco/estação, combate, tutoriais e desafios. Sinais vêm da tela/FRAME e telemetria validada: cena, posição, HP, morte, inimigos, geo e marcos. Tetos são limitados por orçamento e janela.
- `src/hk_agent/scene_change_task.py:274+`: reward diagnóstico de mudança de regime visual e penalidades operacionais/temporais; lê observação e resultados validados.

## 5. Política

- `src/hk_agent/recurrent_sb3.py:99-242`: `MaskableRecurrentPPO`, extrator combinado de observações, LSTM ator/crítico e distribuição categórica mascarável; estado no formato `RNNStates((h,c),(h,c))`.
- `src/hk_agent/first_training.py:58,330-336`: tamanho padrão da LSTM 256, configurável em camadas; rede e otimizador vêm de SB3-Contrib.
- `src/hk_agent/shared_learner.py:977-1080`: agrega rollouts recorrentes, estados LSTM e máscaras entre atores.

## 6. Universal e específico do HK

### Universal (plataforma 2D)
- Intenções de mover, pular, atacar, interagir e escolher duração; pixels e histórico de ações.
- LSTM e máscara sobre intenções disponíveis.
- Recompensas por novidade, progresso, dano, morte, tempo, exploração e travada, quando definidas sem conhecimento de conteúdo exclusivo.

### Específico do Hollow Knight
- Protocolo binário, processo, instalação da DLL, FRAME/ACK, cenas e bootstrap de menu.
- Slots e regras de saves, geo, sombra, amuletos, bancos/estações, objetivos, bestiário e habilidades próprias.
- Mapeamento de botões e sinais específicos do jogo; checkpoints SB3 e caminhos de campanha atuais.

## Limites deste inventário

- O teste de checkpoint pedido em 83.3.2 foi pulado: o runtime/checkpoint local não existe. O teste real e a comparação de comportamento v1 ficam para o PC principal.
