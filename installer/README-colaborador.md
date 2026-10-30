# Instalação do VoidGym (prévia técnica)

O VoidGym usa o seu Hollow Knight da GOG **1.5.78.11833** em cópias isoladas. O instalador nunca distribui o jogo nem altera a instalação original, os saves pessoais ou a chave de registro pessoal.

O treinamento roda localmente em janelas pequenas e sem áudio. Nesta versão, nenhum dado de jogo é enviado para fora do computador. O mapa é opcional e só será oferecido para download após autorização de distribuição; o padrão é não baixar.

## Requisitos

- Windows 10/11 de 64 bits, conta de usuário sem privilégios de administrador;
- Hollow Knight da GOG, versão 1.5.78.11833;
- aproximadamente 30 GB livres;
- Python 3.12 de 64 bits. O bootstrap instala uma versão com URL e hash fixos caso não encontre uma instalação compatível.

## Instalar

Baixe `Instalar-VoidGym.ps1` da Release oficial e execute pelo PowerShell como usuário normal. O script mostra o plano e pede confirmação. Ele grava logs em `%LOCALAPPDATA%\VoidGym\logs`.

Se a política do PowerShell bloquear a execução, use somente para essa sessão:

```powershell
powershell -ExecutionPolicy Bypass -File .\Instalar-VoidGym.ps1
```

## Operar e remover

Quando estiver implementado, o ícone na bandeja abrirá o painel e oferecerá pausa/parada segura. Para remover, use `python -m voidgym.installer desinstalar`; por padrão, o armazém de experiência e os checkpoints serão preservados. `desinstalar --dry-run` mostra os caminhos antes de qualquer remoção.

Esta prévia ainda não pode instalar até que a DLL do bridge tenha uma Release com hash validado. O teste em notebook e a baseline isolada de King's Pass também estão pendentes.
