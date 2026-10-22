# Status da fila v2

## Preparação

- Remoto `Thayrone-L/VoidGym`; branch `v2/fundacao`, baseada em `origin/codex/parallel-instances-isolation`.
- Windows 10 build 26200, Intel Core Ultra 7 258V, 8 núcleos lógicos, 31,5 GiB RAM, sem GPU NVIDIA detectada; C: tinha 697,2 GiB livres no inventário.
- Python 3.12.14 empacotado pelo Codex; `.venv` criado; locks CPU/dev/map instalados.
- Baseline v1 antes da v2: 286 passed, 10 failed, 409 errors, 0 skipped, 8 subtests passed. Falhas/erros dependem de recursos locais do HK.
- `.gitignore` tinha alteração local `agent-prompts/`; preservada junto de `runtime/` e não incluída nos commits.

## Prompts

| Prompt | Status | Commit | Pendente no PC principal com jogo/GPU |
|---|---|---|---|
| 83 | ✅ pronto | `8610ca7` | smoke real da v1 e comparação de checkpoint |
| 83b | ✅ pronto nesta máquina | `5ffd6ba` | executar CUDA com RTX 3060/VRAM e conferir Actions |
| 84 | ✅ pronto nesta máquina | `35de599` | confirmar `STEP_GAME_MS` no jogo real; benchmark local sem concorrência: 3.501/s serial e 3.615/s (6 workers) |
| 85 | 🟡 parcial | `7c31c70` | Treino 8 h/20 M com CUDA na RTX 3060, medir VRAM/throughput, e confirmar STEP_GAME_MS e transferência no jogo real |

## Bloqueios observados

- `mypy src` reporta apenas os dois diagnósticos existentes em `src/hk_agent/map_art.py:27` sobre pyvips; `mypy src/voidgym tools/v2` passa.
- E2E 83b passou em 9,14 s. A suíte integral na validação 83b ficou parada no teste v1 de 20.000 passos e o `faulthandler_timeout=90` disparou; sem resultado final dessa tentativa.
- GitHub CLI 2.102.1 está instalado, mas a keyring ainda devolve token inválido em `gh auth status`; PR e Actions serão atualizados/verificados pelo conector GitHub autenticado.
