# Ferramentas instaladas

Registro local criado em 2026-10-09. Versões devem ser revalidadas antes de atualizações.

| Ferramenta | Versão | Origem | Integridade / observação |
|---|---|---|---|
| Python | 3.12.10 | Python Software Foundation via winget | instalação por usuário |
| Lumafly | 3.3.0.0 | release oficial `TheMulhima/Lumafly` | SHA-256 `BF59F4447E19CCDFE668FFC217439A86E7841D9AFB2A27FF7E2E6EB3CD499F86`; binário sem assinatura Authenticode |
| Hollow Knight Modding API | 1.5.78.11833-77 | release oficial `hk-modding/api` | pacote Windows; SHA-256 publicado e verificado: `bc9f0db3d0916b05cd5a2420bb602fb1b239ce3ff6c289fd84bffb682fb8f1d6` |
| .NET SDK | 10.0.400 | instalação preexistente | usado para harness; o mod final terá alvo compatível com as assemblies do jogo |
| Docker Desktop | 29.6.1 (cliente) | instalação preexistente | daemon não iniciado; não requerido para o MVP |

O Lumafly está em `%LOCALAPPDATA%\Programs\Lumafly\Lumafly.exe`. A instalação automatizada da API foi usada para manter o processo reproduzível; o Lumafly permanece disponível para administração manual futura.

A Modding API foi preservada fora do Git em `third_party/`, aplicada na instalação local configurada e validada pelo `ModLog.txt` do jogo.

Fonte do Lumafly: <https://github.com/TheMulhima/Lumafly/releases/tag/v3.3.0.0>

Fonte da Modding API: <https://github.com/hk-modding/api/releases/tag/1.5.78.11833-77>
