# Arquitetura da v2

## Objetivo

A v2 separa o cérebro do agente das regras de cada jogo. O núcleo trabalha com tela, histórico de
ações e conhecimentos adquiridos pelo jogador. Cada jogo implementa um adaptador. Consulte as
decisões em [`decisoes/`](decisoes/).

## Pastas

- `src/voidgym/core/`: ações universais, observações, semântica, tempo e contrato do adaptador.
- `src/voidgym/envs/`: ambiente Gymnasium que usa qualquer adaptador.
- `src/voidgym/adapters/hollow_knight/`: ponte temporária ao código v1; será portada no prompt 86.
- `src/voidgym/gyms/`: ambientes sintéticos; o ginásio de plataformas entra no prompt 84.
- `src/voidgym/models/`: modelos universais, prompt 85.
- `src/voidgym/training/`: learner e atores da v2, prompts 85–87.
- `src/voidgym/operator/`: integração com painel e bandeja, em etapa futura.
- `tests/v2/`: testes independentes da v2.
- `docs/v2/`: arquitetura, adaptadores e decisões da v2.

## Regras de dependência

O núcleo não depende de adaptadores, ambientes, modelos nem treino. A v1 não importa a v2. O
adaptador de Hollow Knight pode usar imports v1 marcados `V1-SHIM` até o prompt 86.

## Portes planejados

| Peça da v1 | Destino na v2 | Prompt |
|---|---|---|
| Rede e LSTM | `models/` | 85 |
| Learner compartilhado, SIL e imitação | `training/` | 85–87 |
| Bridge e mod do HK | `adapters/hollow_knight/` | 86 |
| Painel, supervisor e bandeja | `operator/` | depois |

Código morto, caminhos `explore_v1`–`explore_v3`, `visual_motion` e hacks de prompts antigos não
serão portados sem uma justificativa de uso.
