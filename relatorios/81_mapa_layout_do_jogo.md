# Etapa 81 — Layout do GameMap e calibração global

## Implementado

- A DLL agora reconhece `HK_AGENT_MAP_LAYOUT_DUMP_REQUEST`, procura o `GameMap` carregado em `Resources` sem instanciar prefab e tenta coletar cenas, área, `SpriteRenderer.bounds`, presença de sprite, versão do jogo e SHA-256 de `Assembly-CSharp.dll`. O JSON de destino é `runtime/map-layout/game-map-layout.json`.
- `tools/dump_game_map_layout.py` valida o marcador de `hk-ai-2`, cria o pedido e abre apenas essa instalação isolada; em timeout solicita fechamento por `WM_CLOSE` e nunca força a terminação.
- O painel valida o documento, mapeia pontos do diário pelos limites da sala ao retângulo do GameMap e aplica uma escala uniforme com inversão vertical e translação até a arte. A calibração global e os deslocamentos por sala são persistidos no cache local da arte. O traço da run continua vermelho e o ícone continua marcando o Cavaleiro. Contornos de sala aparecem temporariamente no assistente.
- O assistente aceita marcos manuais de salas distintas e permite arrastar o contorno selecionado, ajustar com setas, desfazer e voltar ao automático. O botão antigo por sala fica em “Modo avançado”. Quando a calibração global for salva, o render usa o modelo global e indica quais calibrações antigas posicionadas foram substituídas; o JSON antigo é preservado.
- A arte continua no cache local, com a atribuição existente. O layout é carregado apenas pelo painel e não entra na observação da política.

## Medição real e decisão

A build Release foi implantada somente em `hk-ai-2` pelo comando `tools/deploy_bridge.py`; cada implantação atualizou e validou o marcador. O dump foi tentado quatro vezes; as duas primeiras tentativas usaram o destino padrão local incorreto, que foi corrigido para o runtime configurado do painel. Nas duas tentativas seguintes, o status confirmou `waiting-for-loaded-GameMap`; nenhum `game-map-layout.json` foi produzido. O menu mostrou um slot existente “Passagem do Rei”; antes de selecionar, os arquivos `user1*` foram copiados. O slot foi selecionado uma vez na cópia isolada, mas não foi possível confirmar a entrada na cena nem gerar o dump. Os cinco arquivos foram restaurados ao final com SHA-256 idêntico ao backup. Não foi possível calcular erro por sala, decidir com evidência entre Plano A e Plano B, nem capturar o mapa antes/depois. O assistente global está implementado, mas a escolha da estratégia de alinhamento permanece pendente da coleta do GameMap.

Não foi iniciada campanha de treino nem selecionado “Novo Jogo”.

## Validação de código

- Build `HkAgentBridge.Mod` Release contra o `Managed` do Hollow Knight: sucesso, 0 avisos e 0 erros.
- Suíte completa após as alterações Python: **703 passaram, 2 ignorados, 3 avisos conhecidos do Stable-Baselines, 8 subtestes**, em 131,65 s.
- Testes focados de layout, painel e servidor: **31 passaram**.
- `ruff check src tests tools\deploy_bridge.py tools\dump_game_map_layout.py`: passou.
- `mypy src`: passou (56 arquivos).
- `node --check src\hk_agent\control_plane_web\app.js`: passou.
- `git diff --check`: passou.

## Pendente para fechar a etapa

É necessário obter o JSON do GameMap em uma cena carregada. Sem ele, não há base para o relatório de erro por sala, a decisão A/B, nem a validação visual da rota. O soak da etapa 80 também ficou pendente pela instrução de não iniciar campanhas hoje.
