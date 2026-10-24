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

O inventário da v1 está em `relatorios/83_inventario.md`. Ele encontrou vários caminhos legados
que não devem virar dependências do núcleo: versões antigas de recompensa `explore_v1`–
`explore_v3`, a tarefa `visual_motion` de calibração, a tabela de botões brutos com bitmasks,
formatos antigos de disponibilidade de ação e os caminhos de bootstrap/saves de Hollow Knight.
Eles podem continuar na v1 por compatibilidade; só a semântica reutilizável deve ser portada.
Não portar código morto nem atalhos de prompts antigos sem um requisito universal explícito.

## Estado de implementação

O contrato inicial do adaptador, observação e ambiente genérico chegou no prompt 83. Ginásio,
política e campanhas estão previstos nos prompts 84–87 e são incrementos posteriores.

O prompt 83b acrescentou configuração congelada, detecção de hardware, plano de recursos,
armazenamento de experiência, recompensa universal e verificação de regras. O E2E atual cobre
um ciclo sintético sem processos externos; o supervisor de campanhas será completado nos
prompts 86–87.
