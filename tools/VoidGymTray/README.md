# VoidGym Tray

Aplicativo WinForms para Windows, sem dependências externas. O supervisor continua sendo dono do treino; sair do ícone fecha apenas o cliente de bandeja.

## Build e execução

```powershell
dotnet restore tools\VoidGymTray\VoidGymTray.csproj
dotnet build tools\VoidGymTray\VoidGymTray.csproj -c Release
Iniciar-Bandeja.bat
```

O app usa `http://127.0.0.1:8765` e recua para `8766`, autentica mutações com o token de `/api/session`, e consulta `/api/operator` no máximo uma vez a cada cinco segundos. A opção **Iniciar com o Windows** cria um atalho na pasta Inicializar do usuário e vem desligada. Logs do cliente ficam em `artifacts\control-plane\tray-AAAAmmdd.log`.

`dotnet tools\VoidGymTray\bin\Release\net10.0-windows\VoidGymTray.dll --self-test` executa as verificações internas de parsing, escolha de campanha, mutex e escopo de PIDs.
