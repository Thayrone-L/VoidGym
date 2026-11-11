# 86e — avaliação do pré-treino, nível 2 e gabarito Hollow Knight

## Estado

A árvore foi organizada conforme o prompt 86f: o instalador 90 e o rascunho do 87 foram para a branch `v2/instalador`; o PR #8 recebeu em commit separado a correção v1 `7f10902` para promover saves entre volumes por cópia, SHA-256 e `os.replace`. Essa correção foi motivada pela campanha `jogada-1-repeat-1c3b`, que falhou em 21/10/2026 com `OSError: [Errno 28]` quando C: encheu enquanto o runtime já estava em E:.

A avaliação completa do checkpoint `universal-009235520` foi concluída (9.235.520 passos). A Etapa 2 não encontrou defeito no gerador do nível 2; por isso não houve mudança no gerador. A classificação ENEMY foi restringida e a DLL corrigida foi compilada e implantada somente em `hk-ai-1`. A amostra longa de HK ainda não atende aos requisitos de HAZARD e EXIT: a captura chegou a 1.500 quadros, mas ficou em `Tutorial_01`, sem transição e sem HAZARD no gabarito. Uma nova tentativa com a regra corrigida perdeu o canal de captura após 61 quadros. A transferência nessa amostra curta é exploratória, não a medição oficial pedida.

A Etapa 4 de treino **não foi iniciada**. O prompt 86e exige autorização do operador no momento da retomada; o critério foi implementado no código, mas ainda precisa desse “sim”.

## Avaliação lado a lado

Os números de 85 vêm do relatório existente `relatorios/85_eval/tabela.md`; as políticas bobas foram avaliadas nas mesmas salas de teste do nível 2–7. Percentuais das políticas bobas são taxas de sucesso.

| Nível | 85 det. | 85 estoc. | 86e det. | 86e estoc. | Boba RIGHT | RIGHT+JUMP | Aleatória |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 76% | 81% | 93% | 93% | — | — | — |
| 1 | 69% | 78% | 85% | 99% | — | — | — |
| 2 | 62% | 62% | 54% | 59% | 2% | 5% | 1% |
| 3 | 10% | 46% | 86% | 99% | 16,5% | 9,5% | 9% |
| 4 | 1% | 45% | 76% | 90% | 0% | 0% | 19% |
| 5 | 74% | 69% | 95% | 98% | 15,5% | 4,5% | 1% |
| 6 | 18% | 72% | 5% | 94% | 0% | 0% | 0,5% |
| 7 | 0% | 3,3% | 33,3% | 60% | 0% | 0% | 0% |

O checkpoint 86e melhora muito os níveis 0, 1, 3, 4, 5 e 7 no modo estocástico. No modo determinístico, L2 recua 8 pontos contra o 85 e L6 recua de 18% para 5%; no estocástico L6 sobe a 94%. Essa discrepância aponta para sensibilidade à seleção de ação em L6, não para ausência total de comportamento útil. As políticas bobas falham em pelo menos 80% das salas de L2–7, como requerido.

### Percepção semântica balanceada (reavaliação 86f)

A tabela anterior do 86e foi removida porque acumulava os primeiros 5.000 quadros, quase todos do nível 0; seus suportes zero em HAZARD/ENEMY e a comparação com o 85 não eram válidos. A avaliação nova usa 5.000 quadros por split, 625 em cada nível 0–7, coletados com rota fixa independente do checkpoint e espalhados por 100 episódios. Os hashes de amostra coincidem entre 85 e 86e dentro de cada split. Os níveis 5 e 6 têm suporte positivo para ambas as classes obrigatórias.

IoU agregado nos oito níveis:

| Split | Checkpoint | SOLID | HAZARD | ENEMY | BREAKABLE | PLAYER | EXIT |
|---|---|---:|---:|---:|---:|---:|---:|
| Teste | 85 `000697088` | 0,3763 | 0 | 0 | 0 | 0,3786 | 0,1836 |
| Teste | 86e `009235520` | 0,3088 | 0,0045 | 0,0642 | 0,0878 | 0,5132 | 0,1128 |
| Treino | 85 `000697088` | 0,3733 | 0 | 0 | 0 | 0,4706 | 0,1771 |
| Treino | 86e `009235520` | 0,3075 | 0,0051 | 0,0683 | 0,0949 | 0,5988 | 0,1080 |

IoU e suporte nas classes avaliadas com obrigação de cobertura:

| Split | Nível | Suporte HAZARD | IoU HAZARD (85 → 86e) | Suporte ENEMY | IoU ENEMY (85 → 86e) |
|---|---:|---:|---:|---:|---:|
| Teste | 5 | 2.291 | 0 → 0,0102 | 8.414 | 0 → 0,1409 |
| Teste | 6 | 1.546 | 0 → 0,0031 | 9.806 | 0 → 0,1070 |
| Treino | 5 | 2.116 | 0 → 0,0128 | 8.186 | 0 → 0,1750 |
| Treino | 6 | 2.005 | 0 → 0,0033 | 10.770 | 0 → 0,0964 |

O 86e saiu de IoU zero para ENEMY e HAZARD nos dois splits, mas HAZARD continua quase não reconhecido. ENEMY é parecido entre treino e teste, sem sinal forte de decorar apenas o estilo do treino. SOLID e EXIT caíram contra o 85; a correção da amostragem torna essa queda observável, mas não a explica. Os manifests, matrizes e hashes estão em `relatorios/86f_perception/shared_{train,test}/`.

### Inspeção visual dos GIFs

1. **L2:** 165 quadros praticamente sem movimento do agente; ele não transpõe o vão/obstáculo, compatível com a taxa de sucesso de 54%.
2. **L3:** um quadro; a política termina em 9 passos.
3. **L4:** dois quadros; conclui a sala em 13 passos.
4. **L6:** 79 quadros; a rota repete trechos e acumula mortes, terminando travada após 396 passos no caso inspecionado.
5. **L7:** cinco quadros; a política não conclui o conjunto, ficando travada após 321 passos.

A folha de contato revisada está em `relatorios/86e_eval_009235520/gifs/review_contact_sheet.png`; os GIFs individuais dos níveis 2, 3, 4, 6 e 7 estão em `relatorios/86e_eval_009235520/gifs/gifs/`.

## Diagnóstico do nível 2

Foram comparadas 200 salas de teste determinísticas (`seed_base=8600`) com o checkpoint 86e e com o solver nas mesmas salas. O agente venceu 103/200 (51,5%); o solver venceu 200/200, em média 166,11 passos (mín. 62, máx. 288).

| Falha observada | Salas |
|---|---:|
| Caiu no vão | 43 |
| Travou na barreira | 33 |
| Caiu após o vão, antes da barreira | 17 |
| Caiu antes do vão | 3 |
| Caiu na barreira | 1 |
| Tempo esgotado / outra | 0 |

A geometria não encosta nos limites de segurança: largura média do vão = 61,5% do alcance máximo permitido a 85% (intervalo 41,4–82,3%); altura média da parede = 76,6% do limite de 85% (64,7–88,2%). A aproximação da parede é de 3 unidades em todas as 200 salas, também quando o jogador atravessa da direita para a esquerda. A inspeção confirma que a câmera mantém o vão e a barreira visíveis durante a aproximação. A fórmula do nível 2 já deixa 3 unidades de corrida e o solver resolve todas as salas. Portanto, a evidência aponta para política/aprendizado e não para defeito do gerador; o gerador e as taxas de falha das políticas bobas foram mantidos.

Falhas representativas em GIF: `relatorios/86e_level2_diagnosis/gifs/` (cinco tipos). O diagnóstico corrigiu também a seleção da parede no script de análise: ela agora é identificada pela geometria do obstáculo, não pelo primeiro sólido na altura do jogador.

## Gabarito semântico do Hollow Knight

### Regra ENEMY e hierarquia observada

ENEMY exige collider não-trigger no próprio objeto do `HealthManager` ou collider filho com `DamageHero` **local**. `DamageHero` herdado do pai não basta. O novo diagnóstico registra nome, caminho, componentes, camada, `isTrigger`, `HealthManager`, `DamageHero`, `hazardType` e resultado da regra.

- **Crawler Fixed / Crawler 1 / Crawler 3:** `BoxCollider2D`, não-trigger, camada `Enemies`; `DamageHero` e `HealthManager` no corpo. Classificados ENEMY.
- **Hollow Shade(Clone), corpo raiz:** `BoxCollider2D`, não-trigger, camada `Enemies`, com `HealthManager` e `DamageHero`. Classificado ENEMY.
- **Hollow Shade / Terrain Buffer:** `BoxCollider2D`, não-trigger, camada `Terrain Detector`, sem `DamageHero` local; agora excluído de ENEMY.
- **Alert Range New:** `CircleCollider2D`, trigger, camada `Hero Detector`, componente `AlertRange`; excluído.
- **Scream Range / Quake Range / Music Control:** colliders trigger na camada `Hero Detector`; excluídos.
- **Slash:** `PolygonCollider2D`, trigger, camada `Enemy Attack`, com `DamageHero`; excluído pelo requisito `isTrigger == false`.

A suíte semântica passou, o build Release passou e a DLL foi instalada e validada apenas em `hk-ai-1` (`sha256 cffcb879…`). Não houve mudança em `src/hk_agent/` v1.

### Amostras, HAZARD, EXIT e transferencia a seco

As tentativas antigas ficam registradas como historico: a captura longa automatizada ficou em Tutorial_01, e a tentativa adaptativa encerrou em 61 quadros apos `capture_failed`. A captura manual posterior, feita com o operador na instancia isolada `hk-ai-1`, completou a travessia da Passagem do Rei ate Town/Dirtmouth em duas sessoes. O conjunto possui **1.572 quadros unicos**, todos com imagem e rotulo semantico `u4-v1`; **1.532 quadros de jogo** (1.204 em Tutorial_01 e 328 em Town) foram usados na transferencia. Os 40 quadros de menu foram excluidos da inferencia.

| Classe | Quadros com suporte | Celulas rotuladas | Fracao de celulas validas |
|---|---:|---:|---:|
| HAZARD | 271 | 21.196 | 1,04% |
| ENEMY | 163 | 1.180 | 0,06% |
| EXIT | 109 | 3.452 | 0,17% |

A captura manual alcancou a cena Town e inclui as classes exigidas. O gabarito de oito quadros esta em `relatorios/86f_hk_capture_manual/hk_gabarito.png`; os quadros e labels estao no runtime E: e o manifesto combinado e `E:\VoidGymRuntime\86f-manual-capture\final-capture-manifest.json`.

Transferencia a seco feita com o avaliador oficial nos mesmos 1.532 quadros para `universal-000697088` (85) e `universal-009235520` (86e):

| Checkpoint | IoU EMPTY | SOLID | HAZARD | ENEMY | PLAYER | EXIT |
|---|---:|---:|---:|---:|---:|---:|
| 85 `000697088` | 0,4395 | 0,0635 | 0 | 0 | 0,0486 | 0,0015 |
| 86e `009235520` | 0,4766 | 0,0134 | 0 | 0,0012 | 0,0347 | 0,0001 |

O importador existente nao grava `action_history`; o avaliador informa `action_history_available=false` e usa historico nulo. Assim, estes resultados medem transferencia visual sem sequencia de botoes. HAZARD continua com IoU zero nos dois checkpoints; ENEMY e EXIT tambem permanecem muito baixos. A amostra satisfaz o requisito de cobertura, mas os modelos ainda nao demonstram percepcao util dessas classes.

Os saves foram restaurados por SHA-256: o save pessoal coincide com o backup transacional e o save de `hk-ai-1` coincide com o snapshot anterior a captura. A sessao terminou normalmente; nao houve treino.

## Etapa 4 e currículo

O código agora usa a janela móvel real de 100 salas por nível e para somente quando L0–6 ≥70% e L7 ≥30%, após 16 horas, ou quando um nível fica 3 milhões de passos sem melhorar. A curva registrada da campanha 86d está em `relatorios/86e_avaliacao_curriculo.png`; os campos `success_rate_level_6` chegaram a 71,4% porque incluem as salas sorteadas como nível seguinte, não uma janela de treino de L6. O treino 86d parou quando o currículo chegou ao nível 6; L7 não foi treinado nessa campanha.

A retomada da Etapa 4 aguarda a autorização explícita exigida no próprio prompt 86e. Na campanha 86d o commit usado foi 27,5 GB de 33,8 GB; recomendo fechar programas pesados durante o treino, sem fechar nada automaticamente.

## Validação executada

- `pytest -q`: **856 passed, 4 skipped, 3 warnings, 8 subtests passed** (warnings de `stable_baselines3` em testes existentes).
- Ruff: passou.
- mypy: passou em 116 arquivos.
- `git diff --check`: passou (somente avisos Git de conversão CRLF/LF).
- Testes C# semânticos: passaram.
- Build Release da mod: passou, 0 avisos e 0 erros.

## Artefatos principais

- Avaliação completa: `relatorios/86e_eval_009235520/tabela.md`
- Falhas L2: `relatorios/86e_level2_diagnosis/level2_diagnosis.json`
- Gabarito parcial com a regra corrigida: `relatorios/86e_hk_capture_combat/hk_gabarito.png`
- Curva de sucesso por nível: `relatorios/86e_avaliacao_curriculo.png`
- Captura de 1.500 quadros sem transição: `relatorios/86e_hk_capture_longjump/`

## Conclusao

A avaliacao de comportamento, o diagnostico do nivel 2, a captura manual de 1.572 quadros e a transferencia a seco para os checkpoints 85 e 86e estao registrados. A amostra contem HAZARD, ENEMY e EXIT, mas a IoU mostra que a percepcao transferida ainda e muito baixa. A Etapa 4 de treino final nao foi iniciada; o prompt 86e requer autorizacao explicita para essa etapa. Nenhuma campanha foi iniciada nesta verificacao.
