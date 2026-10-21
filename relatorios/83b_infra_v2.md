# Prompt 83b — infraestrutura da v2

## Entregue

- Configuração de campanha congelada, validação com sugestões, migração schema v0→v1, TOML/JSON e hash estável; exemplos de pretrain e HK.
- Detecção de hardware e plano de recursos, overrides validados, lock CUDA separado e setup independente `.venv-cuda`.
- Armazém NPZ atômico em thread de fundo, índice JSONL, filtros, reconstrução de pilha, `extra_*`, retenção de demonstrações/ajudas e amostra de atores.
- Recompensa universal + específica separada, custo temporal contínuo e tetos de positivos por fatia e janela móvel.
- Harness e2e supervisor + learner + 2 atores, retomada e relançamento; workflow Windows/Python 3.12.
- Regras automáticas, ADR 0004 e arquitetura atualizada.

## Medições e validação

- Máquina do branch: Intel Core Ultra 7 258V, perfil observado 8 cores físicos/lógicos, RAM 31,48 GiB (7,84 GiB disponível durante a amostra), Windows, sem CUDA/GPU NVIDIA reportada; disco não detectado pelo processo de benchmark.
- Benchmark CPU com execução abaixo da prioridade normal, 0,25 s: 181,07 convoluções/s. GPU e ganho GPU/CPU indisponíveis; não há `.venv-cuda` nem GPU nesta máquina.
- Perfis artificiais de teste: 8/16 cores, 16 GiB RAM, 12 GiB VRAM → CUDA, 3 instâncias, 8 envs, batch 2048; 4/8 cores, 8 GiB, sem GPU → CPU, 1 instância, 4 envs, batch 512; 16/32 cores, 64 GiB, 24 GiB VRAM → CUDA, 3 instâncias, 24 envs, batch 2048. Perfil local → CPU, 3 instâncias, 2 envs, 1 thread learner, batch 512.
- Custo de append de 1.000 passos: teste `<2 ms/pass` aprovado. Estimativa contratual: 62 KB cru e 10–20 KB comprimido por passo; 1M passos ≈10–20 GB (estimativa, não benchmark de milhão de passos).
- `ruff check src tools tests`: aprovado. `mypy src/voidgym tools/v2`: aprovado (26 arquivos). `pytest tests/v2 -m 'not e2e'`: 24 passed, 2 skipped. `pytest -m e2e`: 1 passed em 9,14 s.
- `mypy src`: permanece com dois diagnósticos antigos em `src/hk_agent/map_art.py:27` (pyvips sem stub e `type: ignore` não aplicável), sem edição v1.
- Uma execução da suíte integral começou e foi interrompida após `faulthandler_timeout=90` no teste v1 `test_continuous_reward_charges_every_step_for_20000_steps_without_a_bench`; a thread estava processando `ExplorationTask.step`. Não há contagem final comparável nessa tentativa. Baseline completa anotada no status foi feita antes do 83b.

## Pendente fora desta máquina

- Executar `tools/v2/check_gpu.py` com lock CUDA na máquina principal, registrar RTX 3060/VRAM e speedup real.
- Verificar Actions do repositório e atualizar PR #8; `gh` não está instalado/autenticado neste ambiente.
- Smoke real do jogo e confirmar `STEP_GAME_MS` no HK.
