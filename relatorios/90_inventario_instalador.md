# Prompt 90 — inventário do instalador do colaborador

**Escopo:** Etapa 0, somente leitura. Não houve mudança nas instalações do jogo nem nos saves pessoais. A RAM está em 97,7%; por isso a campanha 87 não foi iniciada. O runtime de novas campanhas foi redirecionado para `E:\VoidGymRuntime`; o runtime histórico em `C:\Users\thayr\Games\HollowKnight-AI\runtime` permanece intacto.

## Provisionamento e identidade da instância

- `src/hk_agent/instance_install.py:16-32`: define o caminho GOG local, roots de instalação/saves/runtime, perfil `gog-1.5.78.11833`, Modding API `1.5.78.11833-77` e o ZIP empacotado em `third_party/modding-api/`.
- `src/hk_agent/instance_install.py:317-329`: `provision_instance(index, source, instances_root, workspace, resync, sync_mods, bridge_dll, api_archive, editor, free_bytes)`. Rejeita destinos não gerenciados, árvore de origem com links/especiais e espaço menor que o tamanho da instalação + 2 GiB; sincroniza para staging e troca a pasta apenas após validar.
- `src/hk_agent/instance_install.py:357-407`: altera `PlayerSettings.productName` somente no `globalgamemanagers` da cópia para `Hollow Knight AI <n>`, preserva `companyName=Team Cherry`, confere a instalação original antes/depois e grava `.hk-agent-instance.json` com caminhos esperados, SHA-256 do executável, `globalgamemanagers` e DLL, API e versão da ferramenta.
- A identidade separada direciona `persistentDataPath` e a chave Unity para `Hollow Knight AI <n>`; a rotina não deve modificar a instalação GOG original. O launcher injeta as raízes isoladas e verifica o marcador (`src/hk_agent/instance_marker.py`).

## Hashes de versão e Modding API

- Não encontrei hashes **canônicos fixos do jogo** no código/configuração versionada. `installation_hashes` calcula SHA-256 do executável e de `Hollow Knight_Data/globalgamemanagers`, e o marcador verifica se a cópia não mudou; isso prova consistência com aquela instalação, mas ainda não valida a versão suportada contra um hash de perfil pré-fixado.
- Hashes observados agora na instalação local `C:\GOG Games\Hollow Knight` (candidatos para validar a origem do manifesto, não constantes já aprovadas no projeto):
  - `Hollow Knight.exe`: `4d3d0f55afbb249170241d0dc6faf9a5dcf3e1ed427a282ae990bb6e4e3a6913`.
  - `Hollow Knight_Data/globalgamemanagers`: `57ebbd860f452878d82000ed7caa9d40f22436ffcc1e56088eec28683c486efa`.
  - A propriedade `FileVersion` do executável é `2020.2.2.426360` (versão do Unity; não comprova sozinha a versão comercial do jogo).
- `config.example.toml:20-24`, `tools/install-modding-api.ps1:6,76` e `docs/FERRAMENTAS_INSTALADAS.md:9` fixam SHA-256 do ZIP Windows v77: `bc9f0db3d0916b05cd5a2420bb602fb1b239ce3ff6c289fd84bffb682fb8f1d6`. O arquivo local (`third_party/modding-api/1.5.78.11833-77/moddingapi.v77.windows.zip`) conferiu com esse hash.
- URL oficial confirmada: `https://github.com/hk-modding/api/releases/download/1.5.78.11833-77/moddingapi.v77.windows.zip`; release e checksum estão publicados pelo [repositório oficial de releases](https://github.com/hk-modding/api/releases) e pelo manifesto oficial de links em [hk-modding/modlinks](https://github.com/hk-modding/modlinks/blob/main/ApiLinks.xml).
- `tools/install-modding-api.ps1:1-7,70-100,100-167` recebe caminho de arquivo/instalação, exige o hash esperado, valida os conteúdos e cria backups/manifests. Não baixa o pacote por si só; o instalador do colaborador precisará baixar do URL fixo e verificar SHA-256 antes de chamar o instalador.

## DLL do bridge

- `tools/deploy_bridge.py:27-95,121-128`: faz backup da DLL e do marcador nas instâncias selecionadas, copia a DLL, atualiza o marcador por `refresh_instance_bridge_marker` e valida o hash; recusa instalação/jogo em execução.
- Build local: `bridge/HkAgentBridge.Mod/HkAgentBridge.Mod.csproj:1-47`, .NET Framework `net472`; exige `HollowKnightManaged` e referências a `Assembly-CSharp.dll`, `MMHOOK_Assembly-CSharp.dll`, Unity e Newtonsoft.Json do jogo/API. O SDK não redistribui essas DLLs.
- `.github/workflows/code-ci.yml:45-66` compila/roda testes de contrato, mas não compila `HkAgentBridge.Mod`. Não existe `.github/workflows/bridge-release.yml`; distribuir um build pela Release requer resolver como fazer build com referências locais sem publicar assemblies do jogo. O fallback documentado pelo prompt 90 é build local e publicação manual do asset.

## Bandeja, áudio, mouse e janelas

- `tools/VoidGymTray/Program.cs:295-298,352-373`: a bandeja minimiza/restaura apenas as janelas descendentes dos PIDs gerenciados; exibe toast dizendo que a coleta continua pelo supervisor. “Iniciar com o Windows” cria/remove `VoidGym Tray.lnk` na pasta Startup do usuário.
- `relatorios/78_bandeja.md:5-15` registra cliente WinForms sem pacote NuGet extra e self-test aprovado. `relatorios/78_bandeja.md:19-23` registra que a validação visual real do ícone, toast, abrir painel e minimizar/restaurar não foi feita naquela sessão; ritmo de coleta com o jogo minimizado também segue sem medida real.
- `bridge/HkAgentBridge.Mod/AudioMuteGuard.cs:8-12,27-64`: zera `AudioListener.volume` via reflexão a cada frame, sem alterar slider, save ou PlayerPrefs; fica ativo por padrão e só desativa com `HK_AGENT_MUTE=0`.
- `bridge/HkAgentBridge.Mod/PointerInputGuard.cs:15-18` desliga `allowMouseInput` e `focusOnMouseHover` do módulo de input do jogo, reaplicando as flags; não move nem bloqueia o mouse do Windows.
- `src/hk_agent/launch_options.py:4-19` passa `-screen-fullscreen 0`, `-screen-width`, `-screen-height`. `CampaignSpec` usa 960×540 por padrão (`src/hk_agent/control_plane.py:165-166,410-413`). Não usa `-popupwindow`; a janela é reposicionada pelo controle de janelas em `src/hk_agent/window_placement.py`. Portanto não há mudança de PlayerPrefs/registro para definir resolução.

## Arte do mapa (prompts 52/53)

- Os prompts-base 52/53 não existem como arquivos ativos em `agent-prompts/`; restaram recursos em `agent-prompts/relatorios/52_jornada/` e a atribuição no código.
- `src/hk_agent/run_journey.py:32,1120-1159` define página MediaFire estável `https://www.mediafire.com/file/mo42xgq0f4dw8nx/`, permite somente host `mediafire.com`, baixa a arte e a coloca no cache por hash (`cache_map_file`). `src/hk_agent/control_plane.py:2197-2258` inicia/importa o trabalho; `src/hk_agent/map_art.py:13,33-96,124-132` valida SHA, gera cache de tiles e mantém crédito: “Mapa: The Embraced One — hallownest.net · Hollow Knight © Team Cherry”.
- A rota do painel solicita o download/importação explicitamente; não achei gate de consentimento do autor nem manifesto versionado de URL+hash/licença para a distribuição pelo instalador. O instalador deve deixar o recurso desativado até esse consentimento, como determina o prompt 90.

## Configuração necessária para uma instalação nova

- `config.example.toml:8-24,44-74`: template seguro genérico; treinamento, modding, captura e ações privilegiadas desativados. Não basta para uma campanha v1.
- `config.local.toml` (máquina local, treinamento desligado): perfil GOG, caminhos locais do jogo/saves e `parallel_enabled=false`; não deve ser distribuído.
- `config.training.local.toml` (máquina local, ignorado pelo Git): habilita treino, duas instâncias gerenciadas, paralelismo, `training_device=cpu` para reservar a GPU ao jogo e threads do PyTorch. Nesta retomada, o `runtime_root` foi alterado de `C:\Users\thayr\Games\HollowKnight-AI\runtime` para `E:\VoidGymRuntime`; os demais caminhos de jogo e saves isolados continuam locais em C:.
- `config.worker.local.json` é para a fila distribuída de conhecimento e contém `fork_repository=configure-me/VoidGym`; não é requisito do trabalhador v1. O instalador terá que gerar sua própria configuração e não copiar segredos ou caminhos desta máquina.

## Pré-requisitos de branch e próximos passos

A branch atual é `codex/parallel-instances-isolation`, que contém os resultados de 78/80/86. O prompt 90 manda criar `v2/instalador` a partir de `origin/v2/fundacao`, mas essa ref não existe neste clone; nenhum branch de fundação está listado. Antes de publicar o PR do 90, será preciso escolher/recriar a base apropriada ou atualizar as refs remotas.
