# Ferramentas operacionais

## Implantar a DLL do bridge

Toda implantação de `HkAgentBridge.dll` deve usar o comando único abaixo. Ele guarda a DLL e o marcador anteriores, copia o artefato Release, atualiza o marcador pelo helper oficial de `instance_install` e valida o resultado com `load_instance_marker`. Se qualquer validação falhar, restaura os arquivos da instância afetada.

```powershell
.venv\Scripts\python.exe tools\deploy_bridge.py
```

Por padrão, implanta em `hk-ai-1` e `hk-ai-2` a partir de `bridge\HkAgentBridge.Mod\bin\Release\HkAgentBridge.dll`. Use `--bridge-dll`, `--instances` e `--instances-root` apenas quando a operação tiver sido planejada para outros destinos gerenciados. Backups locais ficam em `artifacts\bridge-deploy-backups`.
