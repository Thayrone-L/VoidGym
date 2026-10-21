# Prompt 84 — ginásio procedural

## Implementação

`STEP_GAME_MS=240`: o contrato usa `lease_ms=80` e `action_repeat=3` da v1; 80×3 = 240 ms
de jogo por decisão. A confirmação medida no jogo fica pendente no PC principal. A física roda
em subpassos de 20 ms (12 por ação), com parâmetros aleatórios por sala nas faixas definidas em
`src/voidgym/gyms/platform_gym/params.py`.

Níveis 0–6 geram corredor, degraus, vãos, bordas altas, saída de buraco, perigos/inimigos e
segredos; nível 7 gera conjuntos conectados com porta por habilidade e banco. O solver executa
a mesma física. Os testes cobrem 200 seeds por nível 0–6 e 50 conjuntos do nível 7: 100%
resolvidos. Testes específicos do ginásio: 13 passaram.

Renderização RGB 192×108 usa dez famílias de textura, paletas, personagem, fundo e parâmetros de
câmera sorteados; seeds/famílias de holdout são reservados para teste. O gabarito semântico é
27×48 e sai apenas em `training_labels`. A inspeção visual confirmou quadros variados, com
diferenças nítidas de paleta, textura e silhuetas; o estilo ainda é abstrato, sem animação
detalhada, blur/parallax multicamada ou todos os pós-processamentos opcionais.

## Desempenho e evidências

Benchmark BELOW_NORMAL com 2.000 passos por worker: 1.259 passos/s em um processo e 3.144
passos/s no total com seis workers (multiprocessing Windows); não havia processo v1 para medir.
O resultado single-process ficou abaixo da meta de 1.500; o profiler aponta física e desenho como
principais custos. A amostra curta de 400 passos variou, portanto os 2.000 passos são o número
registrado.

- `relatorios/84_amostras/mosaico.png`
- `relatorios/84_amostras/gabarito.png`
- `relatorios/84_amostras/teste_vs_treino.png`
- `relatorios/84_amostras/solver_nivel_3.gif`, `_4.gif`, `_6.gif`, `_7.gif`

Lint Ruff e mypy do escopo v2 passaram. A suíte da v2 anterior tinha 37 aprovações e 2 skips;
o e2e passou com os adaptadores fake e platform_gym (2 testes). Suíte integral ainda sem
resultado final: teste legado de 20.000 passos excedeu o timeout da tentativa anterior.

**Pendente no PC principal:** validar duração temporal no jogo; rodar CI/Actions e validar
comportamento integrado no ambiente real. Nenhum arquivo `src/hk_agent/` foi alterado.
