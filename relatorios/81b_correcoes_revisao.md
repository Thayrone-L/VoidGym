# 81b — correções da revisão 80/81

## Alterações concluídas

- O dump de `GameMap` agora escolhe a instalação por `--instance` (padrão 2), deriva a pasta de `INSTANCES_ROOT`, consulta o ControlPlane em modo observador, bloqueia campanhas sem identificação confiável e confirma que o jogo isolado e o executável pessoal estão fechados.
- Antes de iniciar o dump, guarda todos os arquivos `user*` da instância com SHA-256; ao sair por sucesso, timeout ou erro, restaura os saves, remove os novos e verifica os hashes. Se a restauração falhar, informa o caminho do backup.
- A DLL de bridge agora captura exceções da coleta em `.error.txt`, desinscreve o hook ao terminar, limita as verificações do pedido a uma por segundo e encerra o jogo após sucesso ou erro.
- Inspeção de metadados do `Assembly-CSharp.dll` da instância 2 confirmou `GameManager.gameMap` como referência `UnityEngine.GameObject`, com componente `GameMap`; o código consulta o objeto instanciado em cena válida, nunca um prefab. O `GameMap` contém a hierarquia direta área → sala. O nome concreto do objeto e a contagem real de salas dependem da coleta em jogo, ainda não realizada.
- O deployment da DLL valida todas as instâncias e os processos antes de copiar, guarda backups verificados, restaura todas as instâncias alteradas diante de falha e confirma os hashes do rollback. O marcador tipado carrega o índice que o painel apresenta.
- Continue agora só aceita as quatro chaves previstas, preserva a contagem de atores se o operador não a alterar e ajusta o intervalo de checkpoint para múltiplo de `rollout_steps × actors`. O seletor acompanha a campanha retomável e a requisição omite `instances` quando a contagem não mudou.
- O snapshot geral deixou de carregar o layout completo e calibrações. A rota `/api/journey/map/status` fornece detalhes sob demanda; o frontend carrega ao exibir o painel do mapa ou abrir a calibração. Layout e calibrações usam cache pela combinação de caminho, mtime e tamanho.
- Python e frontend agora usam os mesmos casos de coordenadas em `tests/fixtures/map_transform_cases.json`.
- `created_unix_ns` do marcador é preservado. O schema v1 é fechado; adicionar `bridge_updated_unix_ns` requer migração de schema, por isso esse campo ficou documentado para uma migração própria.

## Validação

- `dotnet build bridge/HkAgentBridge.Mod/HkAgentBridge.Mod.csproj -c Release` com `HollowKnightManaged` apontando para a instância 2: sucesso, zero avisos.
- Testes focados anteriores às últimas mudanças: 60 passaram.
- Depois da separação do mapa: `test_control_plane.py`, `test_operator_dashboard.py`, `test_map_layout.py`: 115 passaram; Ruff e `node --check` passaram.
- Validação final integrada: `ruff check src tests tools` passou; `pytest -q`: 741 passaram, 3 ignorados, 3 avisos e 8 subtestes passaram, em 128,29 s. `node --check` passou para o frontend e o utilitário; os 3 casos Node do fixture compartilhado passaram.
- Após a última invalidação do cache no frontend, nova verificação: Ruff passou; 157 testes dos componentes alterados passaram em 24,88 s; Node validou os dois scripts e os 3 casos compartilhados; build Release passou sem avisos.
- Node: 3 casos de transformação compartilhados passaram.
- Medição sintética, 400 salas, 100 chamadas por caminho após aquecimento do cache: status detalhado 0,176 ms mediana / 0,209 ms p95; resumo leve 0,109 ms / 0,135 ms. São medições locais de `map_art_status`, não de uma campanha real.
- Consulta somente leitura: `active_count=0`, sem lançamentos ativos; nenhum processo Hollow Knight foi encontrado.

## Coleta real pendente

Não abri o jogo. O dump existente inicia o executável diretamente e ainda não conecta ao bootstrap de menu que carrega um save isolado; fazê-lo assim não garante que `GameMap` esteja instanciado nem permite verificar o estado de save em tela. Para completar a coleta, é necessário reutilizar `run_menu_bootstrap`/`resolve_campaign_bootstrap` do agente v1 com o supervisor e a referência de prontidão existentes, direcionados à instalação e ao save isolados `hk-ai-2`. Se esse bootstrap não puder ser executado como processo de observação, o passo manual é abrir **Hollow Knight.exe dentro de `instances/hk-ai-2`**, escolher Continue no save isolado e aguardar uma sala carregada antes de solicitar o dump protegido. Não usar Steam nem o save pessoal.

Não houve início ou parada de treino, clique em Continue ou deployment da DLL nas instalações reais.

## Entrega

Commit `4b0b95c` (`fix: address bridge and map review findings`) enviado ao branch `codex/parallel-instances-isolation` do PR #8.
