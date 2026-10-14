# Worker distribuído local

Este fluxo prepara uma máquina para treinar a partir de um campeão promovido e
empacota o resultado localmente. Ele não executa `git fetch`, `git push`, login,
download nem upload. Essas operações continuam explícitas e fora do processo de
treino.

## Configuração local

Copie `config.worker.example.json` para `config.worker.local.json` e escolha um
`worker_id` estável, por exemplo `pc-sala`. O arquivo local é ignorado pelo Git.
Os diretórios padrão ficam sob `artifacts/`, que também não é publicado.

`stage_checkpoint` controla se o checkpoint final será copiado para o outbox.
Use `true` nas máquinas próprias. Para contribuições externas, mantenha o
checkpoint em quarentena: um SB3 ZIP usa serialização Python e não deve ser
carregado em uma máquina confiável antes da revisão.

`trusted_ref` identifica a única branch de promoção mantida pelo responsável. O
worker rejeita override para outro ref. O preflight também exige o recibo criado
pelo sync e vincula exatamente geração e hash do campeão; copiar um candidato
externo diretamente para o cache não o torna confiável.

## Ciclo de uma sessão

1. Atualize o ref remoto de forma explícita, se desejar. O worker nunca faz isso.
2. Importe apenas os documentos JSON validados do ref já disponível localmente:

   ```powershell
   .\.venv\Scripts\python.exe tools\worker_cycle.py sync
   ```

3. Coloque o artefato promovido no cache local. Para o SHA-256 `<sha>`, o layout é:

   ```text
   artifacts/distributed/cache/champions/<sha>/manifest.json
   artifacts/distributed/cache/champions/<sha>/checkpoints/checkpoint.zip
   ```

   O preflight confere tamanho, SHA-256, fingerprint de compatibilidade e inventário
   do manifest. Ele não baixa o arquivo e não desserializa o modelo.

4. Congele o campeão para a próxima sessão:

   ```powershell
   .\.venv\Scripts\python.exe tools\worker_cycle.py preflight `
     --session-id campanha-001
   ```

   O JSON retornado contém `resume_checkpoint`. Passe-o ao runner com
   `--resume-checkpoint`. Se a geração for zero, o valor será `null` e a sessão
   começará sem modelo-base. O campeão não muda durante o rollout.

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

O comando fixa `origin/knowledge` em um commit imutável, cria um worktree isolado sob
o outbox, desabilita hooks e produz uma branch `codex/knowledge-*` com exatamente um
commit. Esse commit contém somente `knowledge/index.json` e o JSON sanitizado da
contribuição; checkpoint, recibo de staging, saves e frames não entram nele. Repetir o
comando valida e retorna o mesmo recibo. Uma falha remove o worktree e a branch parciais.

Push e abertura automática de pull request permanecem deliberadamente desabilitados.
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
- Nenhum caminho local, save, frame ou credencial entra na contribuição pública.
- Um cache ausente ou adulterado interrompe o preflight; não há fallback silencioso.
- Candidatos de amigos são dados não confiáveis e devem ser avaliados em ambiente
  descartável, sem tokens ou acesso aos saves pessoais.
