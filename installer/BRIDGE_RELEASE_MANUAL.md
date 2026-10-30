# Publicação manual do bridge

O bridge de Hollow Knight referencia assemblies locais (`Assembly-CSharp.dll`, `MMHOOK_Assembly-CSharp.dll` e módulos Unity). Esses assemblies licenciados não podem entrar no repositório nem nos artefatos do CI; por isso o plano atual é compilar localmente e publicar somente `HkAgentBridge.dll`.

Em um PC Windows com o Hollow Knight GOG 1.5.78.11833 instalado:

1. Confirme que a API v77 está disponível em uma cópia local do jogo; não use arquivos do save pessoal.
2. Defina o caminho de `Hollow Knight_Data\Managed` e compile:

   ```powershell
   $env:HollowKnightManaged = 'C:\GOG Games\Hollow Knight\Hollow Knight_Data\Managed'
   dotnet build .\bridge\HkAgentBridge.Mod\HkAgentBridge.Mod.csproj -c Release
   ```

3. Confira que o build não incluiu assemblies externos:

   ```powershell
   Get-ChildItem .\bridge\HkAgentBridge.Mod\bin\Release
   Get-FileHash .\bridge\HkAgentBridge.Mod\bin\Release\HkAgentBridge.dll -Algorithm SHA256
   ```

4. Publique apenas `HkAgentBridge.dll` como asset de uma Release do VoidGym. Não anexe a pasta `Managed` nem DLLs do jogo.
5. Copie a URL imutável do asset e o SHA-256 calculado para `installer/manifest.json`, mude `bridge_dll.available` para `true` e valide o manifesto. A versão da DLL deve corresponder à branch/tag do código.
6. Recompile e teste `python -m voidgym.installer instalar --dry-run` antes de distribuir o bootstrap.

O build local verificado em 2026-10-25 resultou no SHA-256 `02006742e7b19bd3aeda4a6228c1741f9aefdc1553a50e7e2c01a43a4cdc2768`. Ele ainda não está publicado, portanto `bridge_dll.available` permanece `false` e o instalador não permite instalação real.
