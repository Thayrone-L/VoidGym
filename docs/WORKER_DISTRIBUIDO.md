# Worker distribuído local

Este fluxo prepara uma máquina para treinar a partir de um campeão promovido,
empacota o resultado localmente e, em uma etapa separada, pode publicar somente os
JSONs sanitizados em um fork. O processo de treino não recebe credenciais, não faz
`git fetch`, `git push`, download nem upload.

## Configuração local

Copie `config.worker.example.json` para `config.worker.local.json` e escolha um
`worker_id` estável, por exemplo `pc-sala`. O arquivo local é ignorado pelo Git.
Os diretórios padrão ficam sob `artifacts/`, que também não é publicado.

`stage_checkpoint` controla se o checkpoint final será copiado para o outbox.
Use `true` nas máquinas próprias. Para contribuições externas, mantenha o
checkpoint em quarentena: um SB3 ZIP usa serialização Python e não deve ser
carregado em uma máquina confiável antes da revisão.

`upstream_repository` e `upstream_remote` identificam o repositório canônico. O ref
confiável é sempre `<upstream_remote>/knowledge`; o worker rejeita override para
outro ref. `fork_repository` precisa ser um fork em outra conta e `fork_remote`
precisa ser um remote diferente. O preflight também exige o recibo criado
pelo sync e vincula exatamente geração e hash do campeão; copiar um candidato
externo diretamente para o cache não o torna confiável.

Configure os remotes sem colocar token na URL:

```powershell
git remote add upstream https://github.com/Thayrone-L/VoidGym.git
git remote set-url --push upstream disabled://voidgym-upstream-read-only
git remote add fork https://github.com/SEU-USUARIO/VoidGym.git
```

Use o Git Credential Manager do sistema para autenticar o fork. Não salve PAT em
`config.worker.local.json`, argumentos, variáveis `GH_TOKEN`/`GITHUB_TOKEN` ou URLs.

## Ciclo de uma sessão

1. Atualize o ref remoto de forma explícita, se desejar. O worker nunca faz isso.
2. Importe apenas os documentos JSON validados do ref já disponível localmente:

   ```powershell
   .\.venv\Scripts\python.exe tools\worker_cycle.py sync
   ```

3. Um campeão distribuído nunca pode ser um checkpoint SB3. O único formato promovível
   é uma policy `safetensors`, que não carrega objetos Python. Para o SHA-256 `<sha>`,
   o layout futuro do cache é:

   ```text
   artifacts/distributed/cache/champions/<sha>/manifest.json
   artifacts/distributed/cache/champions/<sha>/policies/policy.safetensors
   ```

   O preflight confere tamanho, SHA-256, fingerprint de compatibilidade e inventário
   do manifest. Ele não baixa o arquivo e não desserializa o modelo.

4. Congele o campeão para a próxima sessão:

   ```powershell
   .\.venv\Scripts\python.exe tools\worker_cycle.py preflight `
     --session-id campanha-001
   ```

   Para geração zero, `resume_checkpoint` e `warm_start_policy` são `null`. Para um
   campeão promovido, `resume_checkpoint` permanece sempre `null` e o recibo aponta
   separadamente para `warm_start_policy`. Não passe esse arquivo para `PPO.load`:
   o carregador tensor-a-tensor ainda será implementado e validado antes de habilitar
   treino a partir de um campeão distribuído. O campeão não muda durante o rollout.

5. Depois de o processo terminar, finalize usando o manifest exato daquela run:

   ```powershell
   .\.venv\Scripts\python.exe tools\worker_cycle.py finalize `
     --session-id campanha-001 `
     --manifest artifacts\training\...\training\manifest.json
   ```

O finalize falha fechado para runs ainda ativas, sem checkpoint retomável,
incompatíveis com o campeão congelado ou cujo checkpoint não pertença ao diretório
da run. A saída idempotente fica em:

```text
artifacts/distributed/outbox/candidates/<bundle_id>/staging.json
artifacts/distributed/outbox/candidates/<bundle_id>/contribution.json
artifacts/distributed/outbox/candidates/<bundle_id>/checkpoint.zip
```

Cada run recebe um bundle imutável próprio; assim, duas runs simultâneas não fazem
read-modify-write no mesmo índice. `staging.json` é um recibo local, não o candidato
público. Depois do upload do checkpoint para uma URL HTTPS imutável, um publicador
único integra `contribution.json` e gera o candidato público com o comando de
exportação do registro de conhecimento. Somente o avaliador confiável promove um
candidato para `knowledge/champion.json`.

Uma contribuição própria pode ser preparada para revisão sem alterar o checkout
principal e sem acessar a rede:

```powershell
.\.venv\Scripts\python.exe tools\worker_cycle.py prepare-publication `
  --bundle-id <bundle_id>
```

O comando usa o `upstream/knowledge` disponível localmente mais recente, cria um
worktree isolado e uma branch inédita com um identificador aleatório de 128 bits. O
commit contém somente `knowledge/index.json` e o JSON sanitizado da contribuição;
checkpoint, recibo de staging, saves e frames não entram nele. O recibo preserva
separadamente o commit usado no treino e o commit usado como base de publicação. Se
outra máquina avançou a branch `knowledge`, uma nova tentativa recompõe a contribuição
sobre a base nova, sem rebase nem force-push.

Depois de revisar o diff local, publique a tentativa apenas no fork:

```powershell
.\.venv\Scripts\python.exe tools\worker_cycle.py publish-fork `
  --bundle-id <bundle_id> `
  --attempt-id <attempt_id>
```

O remote `upstream` tem uma URL de push deliberadamente inválida. O publicador deriva
a URL HTTPS exclusivamente de `fork_repository`, exige que a branch remota ainda não
exista, envia o commit exato sem `--force` e confirma o SHA
remoto. A resposta fornece uma URL de comparação para abrir o pull request
manualmente contra `Thayrone-L/VoidGym:knowledge`. Ele nunca envia para o upstream,
não abre nem mescla PR automaticamente e não promove campeão.

O documento público usa uma allowlist recursiva: ações, sinais e claims são enumerações;
configurações são apenas números/booleanos; identificadores locais e labels são omitidos;
o perfil do jogo é publicado somente como SHA-256; valores livres de versões também são
omitidos. Ainda assim, revise o diff preparado manualmente antes de qualquer publicação.

O `session-id` deve ser único por campanha simultânea. Cada identificador cria um
recibo de preflight imutável, impedindo que uma segunda campanha troque silenciosamente
o campeão-pai congelado pela primeira.

## Garantias e limites

- GitHub não é um barramento de rollout e não combina pesos PPO.
- Cada sessão usa um snapshot fixo do campeão.
- O worker nunca executa código recebido da branch `knowledge`.
- Cada tentativa usa branch remota nova; não há force-push, rebase ou branch reutilizada.
- Nenhum caminho local, save, frame ou credencial entra na contribuição pública.
- Um cache ausente ou adulterado interrompe o preflight; não há fallback silencioso.
- Candidatos de amigos são dados não confiáveis e devem ser avaliados em ambiente
  descartável, sem tokens ou acesso aos saves pessoais.
