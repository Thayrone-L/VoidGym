# Prompt 83 — contrato do adaptador

## Inventário e ambiente

O inventário da v1 está em [`83_inventario.md`](83_inventario.md). A máquina é Windows 10 build 26200, Intel Core Ultra 7 258V, 8 núcleos lógicos, 31,5 GiB RAM, sem GPU NVIDIA; C: tinha 697,2 GiB livres. Python 3.12.14 foi usado do runtime local do Codex.

Comandos usados:

```powershell
& 'C:\Users\thayr\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-training.lock
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r requirements-dev.lock
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r requirements-map.lock
.\.venv\Scripts\python.exe -m pip install -e .
```

`requirements-dev.lock` foi necessário porque o lock de treino não inclui pytest, ruff ou mypy. `requirements-map.lock` instalou o binding pyvips, mas `mypy src` ainda reporta erro preexistente de import/stub em `src/hk_agent/map_art.py:27` (`import-not-found` e `unused-ignore`). `mypy src/voidgym` passa.

## Implementação

- Criado `src/voidgym/` com contrato de adaptador, ações universais, observação 192×108 com pilha RGB de quatro quadros, semântica 27×48, conversão de tempo e `AdapterEnv`.
- `AdapterEnv` constrói o espaço de observação apenas com pixels e histórico; põe `signals` e `training_labels` em `info`.
- `HollowKnightAdapter` compõe o `ExplorationTask` existente. A tabela preserva em ordem as 43 ações de `explore_v4`; ida e volta passa para todas. Os imports v1 estão isolados e marcados `V1-SHIM` para portar no prompt 86.
- Criados os testes falsos, teste de regras de importação e testes de forma 16:9, 4:3 e 21:9.
- Copiados os três documentos de decisão para `docs/v2/decisoes/`; escritos `docs/v2/arquitetura.md` e `docs/v2/adaptadores.md`.
- `STEP_GAME_MS=240`: lease base de 80 ms × `action_repeat=3` no loop diagnóstico de `src/hk_agent/scene_change_probe.py`. Leases especiais (`JUMP_TAP` 50, `JUMP_HOLD` 160, `JUMP_MAX` 450 e foco/carga 1500 ms) continuam sendo escolhidos por `TemporalActionTemplate.resolve_lease_ms`; a confirmação de tempo observado fica para o prompt 84/PC principal.
- Nenhum arquivo em `src/hk_agent/` foi alterado. `.gitignore` mantinha uma alteração local preexistente; ela foi preservada e não faz parte deste commit.

## Validação

- Testes v2: **8 passaram, 2 pulados** (observação real do jogo e checkpoint inexistentes nesta máquina).
- Suíte completa depois da implementação: **294 passaram, 10 falharam, 409 deram erro, 2 pulados**, 8 subtests passaram. Baseline antes de tocar na v2: **286 passaram, 10 falharam, 409 deram erro, 0 pulados**, 8 subtests passaram. A suíte v1 não piorou; os dois skips são os testes do adaptador que precisam do PC principal.
- `ruff check src tools tests`: passou.
- `mypy src/voidgym`: passou. `mypy src`: falha no import opcional `pyvips` da v1, já fora de `voidgym`.
- `git diff --check`: passou.
- Não foi possível executar o teste curto real nem comparar com checkpoint: precisa do jogo, DLL e arquivos do PC principal. Essa validação segue pendente nesse PC.

## Pendências

- PC principal: teste de observação/checkpoint e smoke de 10 minutos pela campanha existente.
- Atualizar o shim em 86 para retirar os imports v1.
- Corrigir o problema preexistente de stubs de pyvips sem alterar a v1 congelada, se a equipe exigir `mypy src` integralmente verde.
