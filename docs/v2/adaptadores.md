# Adaptadores de jogo

Um adaptador traduz observações e intenções entre um jogo e a política universal. A regra de
ouro é que `signals()` e `training_labels()` nunca entram na entrada da política: ficam para
recompensa, telemetria e supervisão auxiliar.

| Intenção | Campos principais |
|---|---|
| mover | `move` -1/0/+1 |
| mirar | `aim`: none/up/down |
| pular | `jump`: instant/short/hold/max |
| atacar / esquivar | `attack`, `dash` |
| habilidade | `special` |
| interagir / menu | `interact`, `menu` |

### Vocabulário implementado em `UNIVERSAL_ACTIONS`

| Índice | Nome | Intenção / duração típica |
|---:|---|---|
| 0 | `NOOP` | Não apertar botão; 80 ms |
| 1–2 | `MOVE_LEFT`, `MOVE_RIGHT` | Movimento horizontal; 80 ms |
| 3–8 | `JUMP_SHORT`, `JUMP_HOLD`, `MOVE_LEFT+JUMP_SHORT`, `MOVE_LEFT+JUMP_HOLD`, `MOVE_RIGHT+JUMP_SHORT`, `MOVE_RIGHT+JUMP_HOLD` | Pulo curto ou médio, com ou sem direção; 50/120 ms |
| 9–11 | `JUMP_MAX`, `MOVE_LEFT+JUMP_MAX`, `MOVE_RIGHT+JUMP_MAX` | Pulo máximo; 450 ms |
| 12–16 | `ATTACK`, `MOVE_LEFT+ATTACK`, `MOVE_RIGHT+ATTACK`, `AIM_UP+ATTACK`, `AIM_DOWN+ATTACK` | Ataque e direção; 50 ms |
| 17–19 | `JUMP+ATTACK`, `MOVE_LEFT+JUMP+ATTACK`, `MOVE_RIGHT+JUMP+ATTACK` | Pulo e ataque combinados; 80 ms |
| 20–22 | `SPECIAL_CAST`, `AIM_UP+CAST`, `AIM_DOWN+CAST` | Feitiço com direção opcional; 50 ms |
| 23 | `FOCUS` | Cura/canalização; 1500 ms |
| 24–28 | `NAIL_ART`, `MOVE_LEFT+NAIL_ART`, `MOVE_RIGHT+NAIL_ART`, `AIM_UP+NAIL_ART`, `AIM_DOWN+NAIL_ART` | Ataque carregado; 1500 ms |
| 29–32 | `DASH`, `MOVE_LEFT+DASH`, `MOVE_RIGHT+DASH`, `AIM_DOWN+DASH` | Esquiva; 50 ms |
| 33 | `SUPER_DASH` | Habilidade carregada; 1500 ms |
| 34 | `DREAM` | Habilidade de sonho; 120 ms |
| 35 | `INTERACT` | Interagir com o cenário; 50 ms |
| 36–41 | `MENU_UP`, `MENU_DOWN`, `MENU_LEFT`, `MENU_RIGHT`, `MENU_SUBMIT`, `MENU_CANCEL` | Navegação e confirmação de menu; 50 ms |
| 42 | `OPEN_MENU` | Abrir o menu do jogo; 50 ms |

Os índices acima descrevem a tabela universal, não substituem os índices próprios de cada
jogo. O adaptador de cada jogo mantém sua própria tabela e implementa as conversões.

### Como Hollow Knight implementa `GameAdapter`

| Método | Implementação do shim |
|---|---|
| `game_id` | Constante `hollow_knight`. |
| `action_table()` | Mantém os 43 nomes de `explore_v4` na ordem da tabela temporal da v1. |
| `to_universal()` / `from_universal()` | Usa o mapa escrito à mão `_HK_TO_UNIVERSAL`; os tempos vêm de `resolve_lease_ms(80)`. O teste cobre ida e volta de todos os índices. |
| `frame_rgb()` | Lê o último `PixelFrame.pixels` do ambiente HK existente. |
| `apply()` | Converte o índice temporal para máscara de botões e chama `HollowKnightEnv.step` com o lease resolvido. |
| `action_mask()` | Usa a disponibilidade mais recente do bridge e a máscara atual da v1. |
| `signals()` | Expõe telemetria do FRAME e do passo (sala, posição, HP, Geo, Shade, inimigos, eventos e disponibilidade), separada da observação. |
| `training_labels()` | Retorna `None`; o HK ainda não fornece esses gabaritos ao shim. |
| `game_time_scale()` | Retorna `1.0`; a v1 não acelera o relógio do jogo. |

O código v1 permanece como está; os imports do shim têm o comentário `V1-SHIM` para a
portabilidade planejada no prompt 86. `AdapterEnv` mantém sinais e rótulos dentro de `info`.

## Adaptar outro jogo

1. Descubra como ler a tela.
2. Descubra como enviar botões.
3. Declare as ações e mapeie para intenções universais.
4. Implemente a máscara de ações.
5. Exponha sinais para recompensa.
6. Exponha gabaritos opcionais, sem alimentar a política.
7. Informe a escala de tempo se disponível.
8. Teste ida e volta da tabela.

Candidatos para depois:

- **Silksong:** provável `src/voidgym/adapters/silksong/`, com mod Unity/BepInEx e contrato de
  leitura de tela/entrada separado do HK.
- **Nine Sols:** provável `src/voidgym/adapters/nine_sols/`, começando por confirmar suporte
  viável a um mod Unity ou outro transporte local.
- **Celeste:** provável `src/voidgym/adapters/celeste/`, usando Everest para integração quando
  o ambiente e as permissões de mod forem confirmados.

Esses jogos são candidatos; nenhum está implementado.
