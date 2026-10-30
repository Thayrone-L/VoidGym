# Prompt 83 — pacote universal e contrato de adaptador

Data da revisão complementar: 2026-10-21

## Inventário da v1

O inventário detalhado está em [83_inventario.md](83_inventario.md). Em `explore_v4`, a tarefa declara `pixels`, `action_history` e `player_knowledge`; porém, `ChannelsFirstObservation` remove `player_knowledge` e entrega à política apenas `pixels` `(12,90,160) uint8` e `action_history` `(3,) int64`. A v1 foi mantida intacta para preservar o comportamento e a compatibilidade dos checkpoints.

As ações v4 são 43, na ordem existente. O mapeamento manual do shim fica em `src/voidgym/adapters/hollow_knight/adapter.py`; os tempos abaixo usam lease-base de 80 ms.

| Índice | Ação HK | Intenção universal | Tempo (ms) |
|---:|---|---|---:|
| 0 | `NOOP` | `NOOP` | 80 |
| 1 | `LEFT` | `MOVE_LEFT` | 80 |
| 2 | `RIGHT` | `MOVE_RIGHT` | 80 |
| 3 | `JUMP_TAP` | `JUMP_SHORT` | 50 |
| 4 | `JUMP_HOLD` | `JUMP_HOLD` | 160 |
| 5 | `LEFT+JUMP_TAP` | `MOVE_LEFT+JUMP_SHORT` | 50 |
| 6 | `LEFT+JUMP_HOLD` | `MOVE_LEFT+JUMP_HOLD` | 160 |
| 7 | `RIGHT+JUMP_TAP` | `MOVE_RIGHT+JUMP_SHORT` | 50 |
| 8 | `RIGHT+JUMP_HOLD` | `MOVE_RIGHT+JUMP_HOLD` | 160 |
| 9 | `JUMP_MAX` | `JUMP_MAX` | 450 |
| 10 | `LEFT+JUMP_MAX` | `MOVE_LEFT+JUMP_MAX` | 450 |
| 11 | `RIGHT+JUMP_MAX` | `MOVE_RIGHT+JUMP_MAX` | 450 |
| 12 | `ATTACK` | `ATTACK` | 50 |
| 13 | `LEFT+ATTACK` | `MOVE_LEFT+ATTACK` | 50 |
| 14 | `RIGHT+ATTACK` | `MOVE_RIGHT+ATTACK` | 50 |
| 15 | `UP+ATTACK` | `AIM_UP+ATTACK` | 50 |
| 16 | `DOWN+ATTACK` | `AIM_DOWN+ATTACK` | 50 |
| 17 | `JUMP+ATTACK` | `JUMP+ATTACK` | 80 |
| 18 | `LEFT+JUMP+ATTACK` | `MOVE_LEFT+JUMP+ATTACK` | 80 |
| 19 | `RIGHT+JUMP+ATTACK` | `MOVE_RIGHT+JUMP+ATTACK` | 80 |
| 20 | `CAST` | `SPECIAL_CAST` | 50 |
| 21 | `UP+CAST` | `AIM_UP+CAST` | 50 |
| 22 | `DOWN+CAST` | `AIM_DOWN+CAST` | 50 |
| 23 | `FOCUS_HOLD` | `FOCUS` | 1500 |
| 24 | `NAIL_ART_CHARGE` | `NAIL_ART` | 1500 |
| 25 | `LEFT+NAIL_ART_CHARGE` | `MOVE_LEFT+NAIL_ART` | 1500 |
| 26 | `RIGHT+NAIL_ART_CHARGE` | `MOVE_RIGHT+NAIL_ART` | 1500 |
| 27 | `UP+NAIL_ART_CHARGE` | `AIM_UP+NAIL_ART` | 1500 |
| 28 | `DOWN+NAIL_ART_CHARGE` | `AIM_DOWN+NAIL_ART` | 1500 |
| 29 | `DASH` | `DASH` | 50 |
| 30 | `LEFT+DASH` | `MOVE_LEFT+DASH` | 50 |
| 31 | `RIGHT+DASH` | `MOVE_RIGHT+DASH` | 50 |
| 32 | `DOWN+DASH` | `AIM_DOWN+DASH` | 50 |
| 33 | `SUPER_DASH_HOLD` | `SUPER_DASH` | 1500 |
| 34 | `DREAM_NAIL` | `DREAM` | 160 |
| 35 | `INTERACT` | `INTERACT` | 50 |
| 36 | `MENU_UP` | `MENU_UP` | 50 |
| 37 | `MENU_DOWN` | `MENU_DOWN` | 50 |
| 38 | `MENU_LEFT` | `MENU_LEFT` | 50 |
| 39 | `MENU_RIGHT` | `MENU_RIGHT` | 50 |
| 40 | `MENU_SUBMIT` | `MENU_SUBMIT` | 50 |
| 41 | `MENU_CANCEL` | `MENU_CANCEL` | 50 |
| 42 | `OPEN_CHARMS` | `OPEN_MENU` | 50 |

## Implementação

- Pacote `src/voidgym/` com `core.actions`, contrato `GameAdapter` e `AdapterInfo`, observação universal com letterbox, semântica, conversão de relógio, `AdapterEnv`, fake adapter e shim HK.
- `UNIVERSAL_ACTIONS` cobre as 43 intenções da tabela. O contrato valida os valores de `special`, `menu`, booleanos, duração e direção; Nail Arts são ataques carregados representados por ataque de 1500 ms.
- O shim mantém os nomes/índices v1; `frame_rgb` lê o último frame; `apply` chama a ação da tarefa v1; a máscara usa a disponibilidade atual; os sinais incluem telemetria do FRAME e eventos do passo; `training_labels()` retorna `None`; `game_time_scale()` retorna `1.0`.
- Sinais e gabaritos continuam dentro de `info`; o ambiente genérico não os acrescenta à observação da política.
- `tests/v2/` cobre dependências, ações, mapeamentos HK, letterbox, divisão sem perda, ambiente falso, performance de 1.000 passos e E2E multiprocesso. O teste de checkpoint só é pulado quando o arquivo não existe.
- `pyproject.toml` usa descoberta automática de pacotes sob `src`; `import voidgym.core` funciona no `.venv` sem `PYTHONPATH`.
- `docs/v2/arquitetura.md` e `docs/v2/adaptadores.md` descrevem a divisão de camadas, limites, tabela universal e checklist para novos jogos.

## Validação

- Testes v2: 33 passaram; 1 foi pulado porque `runtime/shared/jogada-1-repeat-1c3b/checkpoints/round-*.zip` não existe neste checkout.
- Suíte completa v1+v2: 751 passaram, 3 foram pulados, 3 avisos conhecidos do SB3; 125,02 s. Uma primeira execução sob carga teve timeout transitório no E2E de retomada; o E2E isolado passou em 1,84 s e a segunda suíte completa passou.
- `ruff check src tools tests`: aprovado.
- `mypy src`: aprovado em 78 arquivos.
- `git diff --check`: aprovado.
- `import voidgym.core` sem `PYTHONPATH`: aprovado.
- Nenhum arquivo em `src/hk_agent/` foi alterado nesta etapa; não foi necessária mudança de DLL.
- Nenhum treino foi iniciado. O teste real de 10 minutos foi omitido conforme a restrição ativa de não iniciar nem encerrar uma campanha sem pedido específico no momento.

## Diferenças e limites registrados

- O inventário revelou que o contrato de tarefa v4 inclui `player_knowledge`, mas o wrapper usado pelo launcher o remove antes da política. O shim v2 preserva os sinais específicos na saída e não modifica o wrapper v1.
- `STEP_GAME_MS=80` representa o lease atual observado, não uma medição de jogo. O prompt 84 fará a calibração formal.
- O checkpoint da campanha e o estado externo da campanha não estão disponíveis neste checkout para validar espaços ao vivo.
- Os commits de fundação e infraestrutura já presentes são `1dd70a8` (prompt 83) e `2ea978d` (prompt 83b). As correções de contrato e este inventário serão entregues em commit complementar.
