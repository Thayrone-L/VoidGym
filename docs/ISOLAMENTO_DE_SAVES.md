# Isolamento de saves experimentais

## Decisão atual

Hollow Knight `1.5.78.11833` grava no caminho compartilhado do Unity em
`%USERPROFILE%\AppData\LocalLow\Team Cherry\Hollow Knight`. Uma cópia separada do jogo e
um slot diferente não isolam essa escrita. Até existir um hook de caminho comprovado ou
uma conta/VM dedicada, o projeto usa uma única sessão transacional por vez.

O launcher `tools/start-isolated-hollow-knight.ps1`:

1. recusa raízes iguais, aninhadas, com reparse points internos ou com qualquer ancestral que seja reparse point;
2. recusa um processo do jogo já aberto;
3. mantém um lock exclusivo durante toda a sessão;
4. inventaria os saves pessoais por caminho, tamanho e SHA-256;
5. cria uma cópia de backup e confere todos os hashes antes de mover qualquer save;
6. move a pasta pessoal para um nome preservado no mesmo diretório pai;
7. cria no caminho esperado pelo jogo uma junction para a raiz experimental;
8. inicia diretamente o executável, ou um worker cujo processo de guarda seja o executável
   real do jogo, e espera seu encerramento;
9. remove somente a junction cujo destino corresponde exatamente ao manifesto;
10. devolve a pasta pessoal e confere novamente todos os hashes.

O backup validado e o `manifest.json` permanecem para auditoria. A raiz experimental não
é apagada. O launcher não aceita paralelismo.

## Uso planejado

Não executar o jogo, Lumafly ou GOG Galaxy em paralelo. Confirmar antes que nenhuma
ferramenta de nuvem esteja sincronizando a pasta de `LocalLow`.

```powershell
.\tools\start-isolated-hollow-knight.ps1 `
  -GameExecutable "<GAME_ROOT>\Hollow Knight.exe" `
  -PersonalRoot "$env:USERPROFILE\AppData\LocalLow\Team Cherry\Hollow Knight" `
  -ExperimentalRoot "$env:USERPROFILE\Games\HollowKnight-AI\saves\development" `
  -BackupRoot "$env:USERPROFILE\Games\HollowKnight-AI\save-backups"
```

Para um smoke test acompanhado, `-AutoCloseAfterSeconds 20` solicita o fechamento normal
após vinte segundos. Se o processo não encerrar, o launcher falha fechado e mantém a
junction experimental ativa; ele não força o processo nem restaura os saves pessoais sob
um jogo ainda aberto.

Quando um worker for o processo transacionado, `-GuardProcessExecutable` aponta para o
executável real do jogo. O launcher persiste essa identidade no manifesto e a verifica no
preflight, logo antes do start e antes de restaurar. A recuperação manual aplica a mesma
guarda e recusa manifestos novos sem essa informação.

## Smoke da bridge dentro do isolamento

O orquestrador usa as raízes e o perfil de `config.local.toml` por padrão:

```powershell
.\.venv\Scripts\python.exe .\tools\run_isolated_session.py `
  --capture-output .\artifacts\diagnostics\isolated-smoke.png
```

Também é possível fornecer explicitamente `--game-executable`, `--personal-root`,
`--experimental-root`, `--backup-root` e `--game-profile`; o conjunto deve estar completo.
O modo atual não navega menus nem aplica ações. Ele confirma a junction experimental antes
de iniciar o jogo, sobe o supervisor em loopback, aguarda handshake e heartbeat, executa
`PING`, solicita um frame, envia `RELEASE_ALL` e fecha normalmente. Token e identificadores
de autenticação não são gravados em arquivo nem passados na linha de comando: são criados
pelo worker em memória e inseridos somente no ambiente explícito do processo do jogo.

## Recuperação após queda de energia ou encerramento brusco

Uma interrupção impossível de capturar pode deixar o caminho ativo como junction e o
manifesto incompleto. Não abra o jogo. Localize o `manifest.json` mencionado pelo próximo
launcher e execute:

```powershell
.\tools\restore-save-isolation.ps1 `
  -ManifestPath "C:\...\save-backups\hk-save-...\manifest.json" `
  -BackupRoot "$env:USERPROFILE\Games\HollowKnight-AI\save-backups"
```

A recuperação falha fechada se encontrar dois candidatos, um caminho diferente ou hashes
divergentes. Ela nunca sobrescreve uma pasta pessoal existente. Se a pasta preservada não
existir, só recria os saves a partir da cópia cujo inventário ainda corresponda ao manifesto.

## Limites

- Suspensão, queda de energia e término forçado são recuperáveis, mas não atômicos em todo
  o sistema de arquivos; por isso existem duas cópias e um manifesto.
- Abrir manualmente outra instância durante a sessão contorna o lock do launcher e é
  proibido.
- Este mecanismo protege uma instância. Paralelismo continua bloqueado até isolamento por
  usuário/VM ou virtualização de caminho por processo ser comprovado.
- A primeira execução real deve ser acompanhada e comparar os hashes pessoais antes e
  depois; os testes automatizados usam apenas diretórios temporários.

## Primeira validação real

Executada em 2026-10-10 por 20 segundos, sem carregar save e sem automação de menus.
Resultado: status `COMPLETED`, jogo com código 0, raiz pessoal restaurada sem reparse
point e raiz experimental preservada como diretório normal. O manifesto auditável foi
mantido somente no diretório local de backups, fora do Git.
