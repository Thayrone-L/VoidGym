# Fechamento do prompt 86f

## Status

Fechadas as etapas independentes da decisão de travessia manual e do treino final. A árvore de trabalho do PR #8 foi organizada, a percepção v2 foi reavaliada com amostragem balanceada e os artefatos disponíveis do Hollow Knight foram inspecionados. A captura final com saída/EXIT e o treino final ainda estão pendentes pelas razões abaixo.

## Etapa 1 — árvore e correção v1

- O instalador 90 e o rascunho bloqueado do 87 foram movidos para `v2/instalador`; a branch foi publicada em `b07ec18`.
- A correção v1 de transferência entre volumes foi isolada em `7f10902`: cópia temporária, verificação SHA-256 e `os.replace`. Foi motivada pela campanha `jogada-1-repeat-1c3b`, que caiu em 21/10/2026 com `OSError: [Errno 28]` quando C: encheu; o runtime estava em E:.
- A árvore do PR #8 não contém os arquivos do instalador nem o rascunho do 87. Capturas e runtime do jogo permanecem como artefatos locais fora dos commits.

## Etapa 2 — percepção balanceada

O avaliador agora coleta 5.000 quadros por split, distribuídos como 625 por nível (0–7), entre 100 episódios por nível. A rota é fixa e independente do checkpoint. Cada avaliação do par usa os mesmos hashes de amostra; níveis 5 e 6 contêm suporte real para HAZARD e ENEMY. Os manifests registram suporte por nível e candidatos usados.

| Split | Checkpoint | SOLID | HAZARD | ENEMY | BREAKABLE | PLAYER | EXIT |
|---|---|---:|---:|---:|---:|---:|---:|
| Teste | 85 `000697088` | 0,3763 | 0 | 0 | 0 | 0,3786 | 0,1836 |
| Teste | 86e `009235520` | 0,3088 | 0,0045 | 0,0642 | 0,0878 | 0,5132 | 0,1128 |
| Treino | 85 `000697088` | 0,3733 | 0 | 0 | 0 | 0,4706 | 0,1771 |
| Treino | 86e `009235520` | 0,3075 | 0,0051 | 0,0683 | 0,0949 | 0,5988 | 0,1080 |

HAZARD/ENEMY em níveis com suporte:

| Split | Nível | HAZARD: suporte / IoU 85 → 86e | ENEMY: suporte / IoU 85 → 86e |
|---|---:|---:|---:|
| Teste | 5 | 2.291 / 0 → 0,0102 | 8.414 / 0 → 0,1409 |
| Teste | 6 | 1.546 / 0 → 0,0031 | 9.806 / 0 → 0,1070 |
| Treino | 5 | 2.116 / 0 → 0,0128 | 8.186 / 0 → 0,1750 |
| Treino | 6 | 2.005 / 0 → 0,0033 | 10.770 / 0 → 0,0964 |

O 86e passou de zero para algum reconhecimento de HAZARD e ENEMY em treino e teste, mas HAZARD ainda é muito fraco. ENEMY tem resultados próximos entre os splits. SOLID e EXIT ficaram abaixo do 85. A tabela antiga no relatório 86e foi substituída porque media quase só o nível 0 e não podia ser comparada à avaliação do 85.

Dados brutos em `relatorios/86f_perception/shared_test/` e `relatorios/86f_perception/shared_train/`; resumo atualizado em `relatorios/86e_avaliacao_e_gabarito.md`.

## Etapa 3 — gabarito no Hollow Knight

- O log disponível da captura anterior mostra a conexão do bridge encerrada após 61 quadros, mas não contém a exceção interna que causou o fechamento. Portanto, não foi possível provar uma causa mais específica. O capturador passou a tentar até três capturas novamente; se uma resposta de ação falha depois de a ação poder ter sido aplicada, ele recupera com `capture` e não repete a ação. Na nova rota, `retry_events` ficou vazio.
- Foi habilitado o dump de todos os colisores distintos que cobrem a tela, incluindo triggers e todas as camadas, até 8.192 objetos, com hierarquia, componentes, tipo, trigger, layer, `HealthManager`, `DamageHero` e `hazardType`.
- O elemento adicional encontrado na região foi `Hazard Respawn Trigger v2`, com `BoxCollider2D` trigger na camada Default, mas sem `DamageHero`, `HealthManager` ou `hazardType`. A inspeção do comportamento confirmou que ele apenas registra o marcador de respawn em contato com um objeto na camada 9; não é dano. A área dos espinhos aparece no tilemap `Chunk 0 x/Scenemap/TileMap Render Data`, com `EdgeCollider2D` de Terrain. Não foi encontrado componente de dano ambiental nos colisores registrados, então a regra HAZARD não foi ampliada com base em um falso positivo. Testes semânticos C# passaram.
- A rota automatizada permaneceu em `Tutorial_01`, parando em um bloqueio do corredor. LEFT/RIGHT mudaram a coordenada brevemente, mas pulo, alto pulo, ataque, baixo e dash não atravessaram o obstáculo. Não alcançou a Passagem do Rei, Dirtmouth nem EXIT. A cena não mudou e o gabarito ainda não satisfaz a amostra oficial de 1.500 quadros com HAZARD, ENEMY e EXIT.
- Para a opção (b), o operador autorizou a captura jogando na instância isolada `hk-ai-1`, mas ainda vai informar o horário. A execução aguarda essa informação.
- A sessão isolada foi encerrada normalmente. Os saves de `hk-ai-1` foram restaurados a partir do backup em E: e verificados por SHA-256; o inventário dos 100 arquivos de save pessoal coincidiu com o snapshot anterior à sessão. Não foi usado Steam.
- Evidências: `relatorios/86f_hk_capture/`, incluindo `ModLog_20261030_route_attempt.txt`, `route-100.json`, `route-400.json`, `route-700.json`, `highjump-probe.json` e `frames/0167-action-0164.png` (túnel com espinhos) / `frames/0722-action-0717.png` (bloqueio).

## Etapa 4 — treino final

Não iniciado. O prompt 86f exige a frase de autorização do operador para retomar o treino final de `universal-009235520`. Nenhuma campanha foi iniciada por esta execução. Não há checkpoint final da Etapa 4 nem avaliação desse checkpoint.

## Validação

- Suíte completa: **850 passed, 4 skipped, 3 warnings, 8 subtests passed**.
- Testes focados de amostragem e retry de captura: **6 passed**.
- Ruff: passou.
- mypy: passou em `src` (108 arquivos) e nos dois avaliadores de `tools/v2` com os imports locais. A execução ampla `mypy src tools` não é válida por mapear arquivos de `tools` duas vezes; isso foi evitado.
- Quatro harnesses C# passaram. Corrigi também uma asserção obsoleta em `MenuPolicyTests`: a política já aceita `PAUSE` (mask dedicado), então o teste agora verifica o contrato atual.
- Build Release de `HkAgentBridge.Mod`: 0 avisos, 0 erros.
- `git diff --check`: passou; restam apenas avisos de conversão CRLF/LF do Git.

## Pendências para fechar totalmente o 86f

1. Receber o horário em que o operador fará a travessia manual e gerar a captura final com HAZARD, ENEMY e EXIT; em seguida gerar o gabarito e executar a transferência a seco oficial nos checkpoints exigidos.
2. Receber a frase de autorização expressa da Etapa 4, executar o treino final, registrar heartbeat/limite de commit e avaliar o checkpoint final.
3. Atualizar este relatório com os resultados dessas duas etapas pendentes.
