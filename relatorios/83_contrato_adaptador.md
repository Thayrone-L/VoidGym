# Prompt 83 — contrato do adaptador

## Inventário

Consulte `83_inventario.md`. O ambiente atual da v1 usa `pixels` em NHWC empilhado e
`action_history`; a interface universal conserva as chaves, convertendo a imagem para
12×108×192 no ambiente genérico. A v1 não foi alterada.

## Implementação

Criado o pacote `src/voidgym/` com contratos de ações, adaptador, observação, semântica e
tempo; `AdapterEnv`; fake adapter; shim de Hollow Knight; testes isolados; documentação de
arquitetura e checklist de novos jogos. A descoberta automática do setuptools já inclui
subpacotes sob `src`, e `import voidgym.core` funciona no venv.

O shim preserva a tabela temporal de 43 ações em sua ordem existente. O mapa de intenções
está no dicionário `_HK_TO_UNIVERSAL` em `adapters/hollow_knight/adapter.py`; durações vêm de
`resolve_lease_ms(80)`. A observação universal contém imagem e histórico; sinais e gabaritos
permanecem em `info`.

## Validação

- Testes v2: 8 passaram e 1 checkpoint foi pulado; não há checkpoint da campanha no runtime
  versionado/local consultado.
- Suíte total: 711 passaram, 3 foram pulados e 8 subtestes passaram.
- `mypy src`: limpo em 72 módulos; Ruff em `src`, `tools` e `tests`: limpo; `git diff --check`:
  limpo.
- Treino real: não iniciado, de acordo com as regras operacionais.

## Limites

O fixture congelado de nomes, o teste de observação com `FakeBackend` e a ida/volta das ações
estão implementados. Não havia checkpoint da campanha no runtime consultado; o teste ficou
skip. A escala 80 ms veio do lease atual, mas deve ser medida no prompt 84.
