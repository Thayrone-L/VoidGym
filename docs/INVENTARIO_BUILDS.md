# Inventário de builds do jogo

## Perfil candidato `current` — rejeitado para o MVP

Inventariado em 2026-10-09, sem executar ou modificar arquivos.

| Campo | Valor |
|---|---|
| Versão declarada | `1.5.12620` |
| Caminho observado | instalação local não versionada |
| Unity | `6000.0.61f1 (74a0adb02c31)` |
| SHA-256 `hollow_knight.exe` | `7961BBA7243CE322163AD6B0DDFD6EF701413020B92AD0D98502526A396171B0` |
| SHA-256 `UnityPlayer.dll` | `8F2D601F8D3C7F4D29D80BA786C0BE873102BB7E6041EB03964A90B99724D90B` |
| SHA-256 `Assembly-CSharp.dll` | `E9048EF6A633970F735E01EC166D3959F610EAEA7A88D827D48D67B1E5FB87BD` |
| SHA-256 `globalgamemanagers` | `0E9A971C59C072A2DDD5BEDE0B64964F512BDE9596C3583D23E4881BDE35D7F8` |
| Decisão | preservar sem alterações; não usar com Modding API v77 |

### Motivo

A Modding API oficial v77 foi publicada para Hollow Knight `1.5.78.11833`. A build `1.5.12620` migrou para Unity 6, alterando assemblies e internals. Há trabalho experimental para Unity 6 na PR `hk-modding/api#164`, mas ele não é a release oficial estável usada pelo ecossistema principal e possui limitações de compatibilidade.

O pacote v77 não será aplicado a esta build. Ela poderá ser reavaliada futuramente como perfil `current`, em diretório e dados separados.

## Perfil requerido `development`

| Campo | Valor esperado |
|---|---|
| Hollow Knight | `1.5.78.11833` |
| Modding API | `1.5.78.11833-77` |
| Caminho candidato | instalação local configurada por máquina |
| Unity observada | `2020.2.2.426360` |
| SHA-256 `Hollow Knight.exe` | `4D3D0F55AFBB249170241D0DC6FAF9A5DCF3E1ED427A282AE990BB6E4E3A6913` |
| SHA-256 `UnityPlayer.dll` | `5BF3B31C012E9AA0B87CD70B9948F19DA7C9E72C51810E2334BE81300317D037` |
| SHA-256 `Assembly-CSharp.dll` observado | `FCC01E0DF1B841A8FAF6B5E39F27030F63204AD02F430B3DEFA262134AE4E8A0` |
| Estado | aprovado; Modding API v77 instalada e validada |

### Resíduos encontrados e tratados

A pasta GOG contém `Hollow Knight_Data\Managed\Mods` com 95 arquivos, aproximadamente 102 MiB e 22 diretórios de mods, além de DLLs MonoMod/MMHOOK no diretório `Managed`. Entre os mods estão Randomizer, ItemChanger, HKMP, Benchwarp, QoL e dependências.

Esses arquivos pareciam anteriores à instalação GOG atual. A pasta `Mods` foi retirada do caminho ativo e arquivada com manifesto; os arquivos substituídos pela API também foram preservados no mesmo backup recuperável.

### Instalação e smoke test

- API instalada: `1.5.78.11833-77`.
- 11 arquivos instalados e verificados contra o pacote oficial.
- 95 arquivos legados movidos para backup recuperável.
- Backup: diretório local não versionado.
- O jogo permaneceu ativo por 20 segundos e encerrou com código 0.
- `ModLog.txt` confirmou `Modding API: 1.5.78.11833-77` e nenhum mod carregado.
- Nenhum arquivo `user*.dat*` mudou durante o teste.
- A ausência do serviço GOG Galaxy gerou aviso de autorização, sem impedir a inicialização.

### Ponte local

- Mod carregado: `HkAgentBridgeMod 1.0.0.0`.
- DLL instalada em `Hollow Knight_Data\Managed\Mods\HkAgentBridge\HkAgentBridge.dll`.
- SHA-256 da versão atual: `A651D29DA9EF5C7862C5C9CACACFFA12D8A023E54FC815C476489B43CD9715D1`.
- Handshake autenticado validado contra o supervisor Python.
- Versão reportada pelo processo real: `1.5.78.11833`.
- Cinco heartbeats foram recebidos no teste de sessão persistente.
- `PING`, `APPLY_ACTION` com lease de 100 ms e `RELEASE_ALL` foram confirmados no processo real.
- Um dispositivo InControl virtual foi anexado e removido internamente, sem depender do foco do Windows.
- O lease máximo aceito é 1 segundo; expiração, desconexão e fechamento zeram todas as ações.
- `CAPTURE_ONCE` retornou um PNG RGB8 de `600x338`, com 4.948 bytes, capturado apenas do framebuffer visível.
- SHA-256 do primeiro frame: `C7DFB1549F427EE0686BA3F4925191EC89BC8140023B8EB5612FB549F62618BB`.
- O primeiro ensaio real de isolamento de saves terminou com status `COMPLETED` e código 0; o caminho pessoal foi restaurado como diretório normal e conferido contra o inventário SHA-256.
- Jogo encerrou com código 0 e nenhum `user*.dat*` foi alterado.

Fontes:

- <https://github.com/hk-modding/api/releases/tag/1.5.78.11833-77>
- <https://github.com/hk-modding/api/issues/179>
- <https://github.com/hk-modding/api/pull/164>
