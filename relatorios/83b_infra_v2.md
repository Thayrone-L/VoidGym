# Prompt 83b — infraestrutura da v2

## Implementado

- Configuração de campanha frozen, com defaults, validação, sugestões, migração inicial e hash
  estável; dois exemplos TOML em `configs/v2/exemplos/`.
- Detecção tolerante de hardware e plano de recursos, com quatro perfis de teste e comparação
  conservadora com a recomendação de instâncias da v1.
- Escritor assíncrono de blocos NPZ, leitor com filtros, retenção e recompensa universal
  separada da recompensa específica do jogo.
- Caminhos centrais em `core/paths.py`, regras de código testadas e workflow Windows para a v2.
- `tools/v2/run_learner.py` e `run_actor.py` mínimos para o ciclo falso de rollout → update →
  publicação/importação de política.
- E2E com processos separados: dois atores, três rodadas, falha/relançamento de um ator,
  checkpoint real em JSON, encerramento limpo e retomada em uma nova rodada.
- Retenção volta ao alvo de 70% do orçamento e protege o mínimo de 10% dos blocos por dia e cena;
  demonstrações e `help_resolved` permanecem protegidos.

## Pendente

- Os executáveis são harness mínimo falso; integração com learner/modelo de produção fica nos
  prompts 85–87. Nenhum treino real foi iniciado.
- O lock CUDA é versionado sem hashes de pacote, como o lock CPU existente.
- A estimativa 62 KB cru e 10–20 KB comprimidos por passo veio do planejamento do projeto; não
  foi medida com rollouts completos nesta etapa.

## Hardware e benchmark local

- GPU: NVIDIA GeForce RTX 3060, 12,00 GB VRAM, capacidade 8.6; driver NVIDIA 617.14 / CUDA UMD
  13.4. CPU: 8 núcleos físicos e 16 lógicos; RAM total 15,80 GB (livre ~4,81 GB na medição).
- O lock CPU mantém PyTorch 2.14.1; o `.venv-cuda` isolado foi instalado com PyTorch 2.14.1+cu130
  e `psutil==7.2.2`. O `.venv` usado pela v1 não foi alterado.
- Plano detectado: CUDA, 2 instâncias, 10 ambientes de ginásio, batch 2048 e 2 threads learner.
  O orçamento de armazenamento fica limitado a 15,52 GB, pois há ~19,40 GB livres no disco;
  alerta de menos de 20 GB livres permanece.
- Microbenchmark de convolução, 10 s por dispositivo: CPU 12,1 ops/s; RTX 3060 423,1 ops/s;
  aceleração ~34,97×. Esse resultado é apenas do microbenchmark, não estima passos/s de treino.

Planos sintéticos (todos mantêm o contrato do modelo):

| Perfil | Dispositivo | Instâncias | Env. ginásio | Batch | Orçamento |
|---|---:|---:|---:|---:|---:|
| Operador (8/16 cores, 16 GB, RTX 3060 12 GB) | CUDA | 2 | 10 | 2048 | 100 GB |
| Notebook (4/8 cores, 8 GB, sem GPU) | CPU | 1 | 4 | 512 | 64 GB |
| PC forte (16/32 cores, 64 GB, GPU 24 GB) | CUDA | 3 | 24 | 2048 | 100 GB |
| PC medido (8/16 cores, 15,80 GB, RTX 3060) | CUDA | 2 | 10 | 2048 | 15,52 GB |

- Escrita do store: 1.000 appends com média 0,00157 ms, p50 0,00140 ms, p95 0,00260 ms e
  máximo 0,03030 ms por passo; o bloco ficou com 62.229.173 bytes (60,77 KiB/passo) usando
  quadros RGB aleatórios como pior caso de compressão. A contagem completa após fechar o writer
  foi confirmada. Esse pior caso não valida os 10–20 KB esperados para cenas reais; continua
  pendente medir rollouts de jogo/gym.

## Validação

- Suíte completa final: 729 passaram, 3 ignorados, 3 avisos, em 125,37 s.
- Suíte v2: 26 passaram, 1 ignorado; E2E multiprocesso 1 passou em 2,23 s. Ruff (incluindo
  B/C901/I/N, complexidade máxima 12) em `src`, `tests`, `tools`, mypy em `src/voidgym` e
  `tools/v2`, e `git diff --check` passam.
