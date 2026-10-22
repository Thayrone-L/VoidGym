# Arquitetura do VoidGym v2

A v2 separa uma política reutilizável dos detalhes de cada jogo. `core/` define os contratos,
`adapters/` conecta jogos, `envs/` expõe ambientes Gymnasium, `gyms/` fornece ambientes
sintéticos, `models/` mantém as redes e `training/` organiza campanhas. `operator/` será a
interface de operação. Os testes da v2 ficam isolados em `tests/v2/`.

O núcleo não importa adaptadores nem código v1. Apenas `adapters/hollow_knight/` pode usar
`hk_agent`, com marcação temporária `V1-SHIM`, a ser portada no prompt 86. A v1 continua
independente e não importa `voidgym`.

| Peça da v1 | Destino da v2 | Prompt |
|---|---|---|
| rede/LSTM | `models/` | 85 |
| learner, SIL e imitação | `training/` | 85–87 |
| bridge e mod HK | `adapters/hollow_knight/` | 86 |
| painel, supervisor e bandeja | `operator/` | posterior |

Não portar código morto, compatibilidade de exploração antiga nem atalhos específicos do HK.
O inventário detalhado da v1 está em `relatorios/83_inventario.md`.

## Estado de implementação

O contrato inicial do adaptador, observação e ambiente genérico chegou no prompt 83. Ginásio,
política e campanhas estão previstos nos prompts 84–87 e são incrementos posteriores.
