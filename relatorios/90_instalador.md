# Prompt 90 — instalador de colaborador (andamento)

**Branch:** `v2/instalador`, baseada no commit local `b4f634e` (`v2/fundacao`). No início não havia `origin/v2/fundacao`; a base local foi criada a partir do commit de `codex/parallel-instances-isolation`, que já contém os prompts 78, 80 e 86.

## Inventário e estado atual

O inventário completo está em [`90_inventario_instalador.md`](90_inventario_instalador.md). O provisionador existente é `src/hk_agent/instance_install.py::provision_instance`: valida hashes da origem, rejeita links/reparse points, copia para `hk-ai-N`, troca `productName` por UnityPy, instala Modding API v77 e bridge, cria o marcador e verifica que a origem não mudou. Usa por padrão `C:\GOG Games\Hollow Knight`, instâncias em `C:\Users\thayr\Games\HollowKnight-AI\instances`, dados pessoais sob LocalLow e runtime em `C:\Users\thayr\Games\HollowKnight-AI\runtime`; `instances.runtime_root` pode ser redirecionado pela configuração v1.

Hashes GOG 1.5.78.11833 medidos localmente e registrados no manifesto: executável `4d3d0f55afbb249170241d0dc6faf9a5dcf3e1ed427a282ae990bb6e4e3a6913`, `globalgamemanagers` `57ebbd860f452878d82000ed7caa9d40f22436ffcc1e56088eec28683c486efa`. A Modding API oficial v77 usa a release `1.5.78.11833-77`; o pacote local conferido tem SHA-256 `bc9f0db3d0916b05cd5a2420bb602fb1b239ce3ff6c289fd84bffb682fb8f1d6`.

O bridge referencia assemblies locais de Hollow Knight/Unity; estes não devem ser enviados ao CI. O build Release local foi repetido com sucesso, usando-os somente como referências: 0 avisos, 0 erros. DLL: `bridge/HkAgentBridge.Mod/bin/Release/HkAgentBridge.dll`, SHA-256 `02006742e7b19bd3aeda4a6228c1741f9aefdc1553a50e7e2c01a43a4cdc2768`. `tools/deploy_bridge.py` faz backup das DLLs e marcadores, recusa jogos em execução e valida/rollback de cada implantação.

O mapa é opcional e sem autorização de distribuição: `distribution_approved=false`; não há URL/hash configurados e o instalador não deve baixá-lo. A baseline v1 está resumida em `87_comparacao_v1_v2.md`: os dados disponíveis começam em `Tutorial_01`, sem saída de cena; não há baseline isolada conhecida na Passagem do Rei.

## Implementado

- Esqueleto `src/voidgym/installer/`: manifesto validado, logger UTF-8, UI em português, estado JSON atômico, adaptador Windows, download HTTPS com publicação somente após SHA-256 e protocolo de passos idempotentes.
- Remoção limita caminhos registrados ao diretório VoidGym ou a `Hollow Knight AI N` sob LocalLow; chaves de registro e atalhos também têm limites de segurança.
- `--dry-run` não cria logs nem diretórios; informa a pendência do bridge. O manifesto só libera instalação quando URL e SHA da DLL estiverem publicados.
- README está marcado como prévia técnica, sem afirmar que a instalação esteja pronta.
- Testes para esquema/hash do manifesto, estado, limites de remoção e downloads com hash certo/errado.

## Validação executada

- Ruff: passou.
- Mypy dos módulos do instalador: passou.
- `pytest tests/v2/installer -q`: 12 passaram.
- `compileall` nos módulos e testes: passou.
- `instalar --dry-run` com diretório temporário: mostrou o plano; o diretório não foi criado.
- Build Release local do bridge: passou, sem alterar a instalação do jogo.

## Pendências antes de concluir o prompt 90

1. O fluxo de instalação real ainda não executa os 12 passos; `instalar` está bloqueado até publicar a DLL. `atualizar` também não integra parada/retomada segura de campanha.
2. Publicar a DLL em uma Release, fixar URL/SHA no manifesto e testar o download. A publicação externa ainda não foi feita.
3. Completar `Instalar-VoidGym.ps1` e criar lock com hashes para `pip --require-hashes`; `requirements-training.lock` atual fixa versões, mas não tem hashes.
4. Completar detecção Steam/GOG, provisionamento/configuração de instâncias, save base isolado, janela/mudo, tray/atalhos, smoke de 60 s e atualização segura.
5. Executar dry-run e instalação real no notebook Core Ultra 7 e registrar o resultado. Este PC principal RTX 3060 não substitui o teste solicitado no notebook.
6. Obter consentimento do autor para distribuir a arte do mapa; permanece desativada por padrão.
7. Fazer validação geral, `git diff --check`, commit, push e PR rascunho para `v2/fundacao` quando o conteúdo estiver pronto.

## Campanha (prompt 87, em andamento)

Após o operador liberar memória, a campanha `v2-hk-1` foi iniciada em `Tutorial_01`. A primeira tentativa revelou incompatibilidade do bridge; o grupo foi parado cooperativamente e as duas cópias gerenciadas receberam o bridge Release atual com backups. A mesma versão foi relançada e os atores começaram a gravar rollouts. O monitor notificará somente falha, ausência de progresso por 10 min ou conclusão. Nenhum aplicativo alheio foi encerrado e o PC não foi desligado.
