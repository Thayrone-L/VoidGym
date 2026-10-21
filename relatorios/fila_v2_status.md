# Status da fila v2

## Preparação

- Repositório remoto confirmado: `Thayrone-L/VoidGym`.
- Branch: `v2/fundacao`, criada sobre `origin/codex/parallel-instances-isolation`.
- Ambiente: Windows 10 (build 26200), Intel Core Ultra 7 258V, 8 núcleos lógicos, 31,5 GiB RAM; sem GPU NVIDIA; disco C: com 697,2 GiB livres no inventário.
- Python 3.12.14 (runtime empacotado pelo Codex); `.venv` criado com esse executável. Lock de treino e lock de desenvolvimento instalados.
- Linha de base v1: `pytest --basetemp .test-tmp/pt83-baseline -o faulthandler_timeout=90 -q --tb=no`: 286 passed, 10 failed, 409 errors, 0 skipped; 8 subtests passed. Erros/falhas pré-existentes foram registrados antes de mudanças da v2.
- `.gitignore` já tinha alteração local (`agent-prompts/`); preservada junto da entrada `runtime/` existente na branch destino.

## Prompts

| Prompt | Status | Commit | Pendente no PC principal com o jogo |
|---|---|---|---|
| 83 | ✅ pronto nesta máquina | pendente de commit | smoke real da v1 e comparação de checkpoint no PC principal. `mypy src` aponta somente o import opcional pyvips preexistente; `mypy src/voidgym` passa. |
| 83b | pendente | — | execução do check CUDA no PC principal. |
| 84 | pendente | — | confirmar `STEP_GAME_MS` no jogo real. |
| 85 | pendente | — | transferência/avaliação do checkpoint no HK. |

## Bloqueios / tentativas

- Prompt 83: `mypy src` mantém dois diagnósticos em `src/hk_agent/map_art.py:27` (pyvips sem stub e ignore não aplicável). Instalado `requirements-map.lock`; o aviso persiste. Nenhuma mudança em `src/hk_agent/`. `mypy src/voidgym` está verde e a linha de base integral da suíte não piorou.

- O primeiro `pytest` alcançou 100%, mas o sandbox bloqueou a limpeza do diretório temporário. Repetido com execução fora do sandbox e basetemp dentro do workspace para obter as contagens.
