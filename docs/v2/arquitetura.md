# Arquitetura da v2

## Objetivo

A v2 separa o cérebro do agente das regras de cada jogo. O núcleo trabalha com tela, histórico de ações e conhecimentos adquiridos pelo jogador. Cada jogo implementa um adaptador. Consulte as decisões em [`decisoes/`](decisoes/).

## Pastas

- `src/voidgym/core/`: ações, observações, semântica, contrato, configuração validada, caminhos, hardware, recompensa e armazenamento de experiência.
- `src/voidgym/envs/`: ambiente Gymnasium que usa qualquer adaptador.
- `src/voidgym/adapters/hollow_knight/`: ponte temporária ao código v1; será portada no prompt 86.
- `src/voidgym/gyms/`: ambientes sintéticos; o ginásio de plataformas entra no prompt 84.
- `src/voidgym/models/`: modelos universais, prompt 85.
- `src/voidgym/training/`: learner e atores da v2, prompts 85–87.
- `src/voidgym/operator/`: integração com painel e bandeja, em etapa futura.
- `tools/v2/`: setup CUDA e processos mínimos de ator, learner e supervisor para o teste ponta a ponta.
- `tests/v2/`: testes independentes da v2 e ciclo integrado sem jogo.
- `docs/v2/`: arquitetura, regras, adaptadores e decisões.

## Configuração e recursos

`core/config.py` concentra `CampaignConfig`, validação, migração, hash estável e serialização TOML/JSON. `core/paths.py` determina o runtime fora do repositório. `core/hardware.py` detecta CPU, RAM, CUDA e disco e gera `ResourcePlan`; as três configurações de exemplo ficam em `configs/v2/exemplos/`. CUDA tem lock e ambiente separados e requer validação na máquina com GPU.

## Dados e recompensa

`core/experience_store.py` grava blocos NPZ comprimidos atomicamente por uma thread de fundo e publica metadados em índice JSONL. Leitores filtram fonte, cena, tags, contrato e datas e recompõem pilhas. A retenção protege demonstrações e episódios com `help_resolved`. `core/reward.py` mantém valores universais e do jogo separados, preserva custo temporal e limita ganhos por fatia de episódio e janela móvel.

## Operação e validação

`tools/v2/run_supervisor.py` inicia learner e dois atores, relança ator encerrado e permite retomada. O teste `@pytest.mark.e2e` percorre rodadas, falha de ator, parada e retomada usando adaptador falso. `.github/workflows/v2-tests.yml` executa lint, tipos, testes unitários e e2e em Windows/Python 3.12.

## Regras de dependência

O núcleo não depende de adaptadores, ambientes, modelos nem treino. A v1 não importa a v2. O adaptador de Hollow Knight pode usar imports v1 marcados `V1-SHIM` até o prompt 86.

## Portes planejados

| Peça da v1 | Destino na v2 | Prompt |
|---|---|---|
| Rede e LSTM | `models/` | 85 |
| Learner compartilhado, SIL e imitação | `training/` | 85–87 |
| Bridge e mod do HK | `adapters/hollow_knight/` | 86 |
| Painel, supervisor e bandeja | `operator/` | depois |

Código morto, caminhos `explore_v1`–`explore_v3`, `visual_motion` e hacks de prompts antigos não serão portados sem justificativa.
