# Conhecimento distribuído

O diretório `knowledge/` é o único conteúdo de treino desenhado para circular pela
branch pública `knowledge`. Cada contribuição contém o contrato do treino, métricas,
hashes dos checkpoints e um fingerprint portátil de compatibilidade. Ele não contém
saves, imagens, checkpoints, executáveis, caminhos locais, credenciais ou identidade
da máquina.

## Fluxo por máquina

Antes de iniciar uma rodada, atualize as referências Git de forma normal e importe a
branch sem fazer checkout nem executar seu conteúdo:

```powershell
git fetch origin knowledge
hk-agent knowledge sync --ref origin/knowledge
hk-agent knowledge validate
```

Depois da rodada, exporte somente o manifesto final:

```powershell
hk-agent knowledge export --manifest artifacts/training/<run>/training/manifest.json
hk-agent knowledge validate
```

O export cria `knowledge/contributions/<id>.json` e atualiza `knowledge/index.json`.
Revise o diff, faça commit e push seguindo o fluxo normal do repositório. A CLI não
usa tokens, não faz `fetch`, `checkout`, `merge`, `commit` ou `push` automaticamente.
Em um repositório público, proteja a branch e integre contribuições de amigos por pull
request.

O exportador aceita apenas a estrutura local produzida pelo próprio agente, mas não pode
determinar se uma string arbitrária escrita no contrato contém informação privada. Nunca
exporte manifestos recebidos de terceiros e sempre revise o JSON sanitizado antes de
publicá-lo. A automação de push deve permanecer desabilitada até existir uma allowlist
fechada para todas as versões de contrato aceitas.

## Candidatos e campeão

`knowledge/champion.json` é o ponteiro canônico para o melhor checkpoint já avaliado.
Antes do treino, registre sua geração e SHA-256 canônico:

```powershell
hk-agent knowledge champion-validate
```

Depois de exportar a contribuição e publicar o checkpoint como um asset HTTPS imutável,
crie o manifesto do candidato. A URI não pode conter credenciais, query string ou
fragmento; o comando registra metadados, mas não baixa nem abre o checkpoint:

```powershell
hk-agent knowledge candidate-export `
  --contribution knowledge/contributions/<id>.json `
  --worker-id maquina-1 `
  --artifact-uri https://github.com/owner/repo/releases/download/run-1/model.zip `
  --artifact-sha256 <sha256> --artifact-size-bytes <bytes> `
  --trust-tier external-untrusted `
  --parent-generation <geracao> --parent-sha256 <sha-do-campeao>
```

O candidato sai em `knowledge/outbox/candidates/` como arquivo imutável. Cada worker
deve manter seu próprio checkout/outbox; não execute exportações concorrentes contra o
mesmo `index.json`, pois a atualização local do índice não é um banco transacional.

Promoções estão temporariamente desabilitadas. O comando abaixo permanece reservado e
falha fechado até existir um recibo verificável do avaliador; hashes fornecidos como
argumentos e o tier autodeclarado pelo produtor não concedem autoridade:

```powershell
hk-agent knowledge champion-promote `
  --candidate knowledge/outbox/candidates/<id>.json `
  --expected-generation <geracao> --expected-sha256 <sha-do-campeao> `
  --evaluation-report-sha256 <sha-do-relatorio> `
  --evaluation-suite-sha256 <sha-da-suite> --evaluator-commit <commit-40-hex>
```

A futura promoção deverá ser idempotente, usar compare-and-swap e rejeitar candidato
baseado em outro campeão ou com fingerprint incompatível. O formato `sb3-zip` é apenas
catalogado: arquivos SB3 externos podem conter objetos serializados e nunca devem ser
desserializados no processo ou na máquina confiável de promoção. Avalie contribuições
de terceiros em ambiente descartável e isolado.

Os tiers `external-untrusted` e `quarantined` não podem ser promovidos enquanto não
existir uma attestation fechada e verificável do avaliador. Além disso, nenhum
`sb3-zip` é promovível, inclusive quando foi produzido numa máquina própria: o ZIP
pode conter pickle executável. O registro campeão aceita apenas uma policy
`safetensors`; checkpoints SB3 continuam estritamente locais.

Uma quarentena futura deverá converter o artefato dentro de uma VM descartável para
`safetensors`, destruir a VM e produzir um recibo que vincule candidato, hashes,
imagem da VM, suite, relatório e commit do avaliador. Até esse recibo existir, a
classificação `quarantined` é apenas catálogo e não concede autoridade.

## Compatibilidade e limites

O fingerprint considera contrato de ações/recompensas, perfil do jogo, origem lógica
da campanha, dimensões da observação e hashes dos componentes. Caminhos absolutos são
descartados, então instalações em diretórios diferentes podem ser compatíveis.

Uma contribuição é imutável e endereçada pelo hash de seu conteúdo. O sincronizador
lê objetos exclusivamente com `git show`, limita tamanho, exige caminhos canônicos,
valida todo o lote antes de gravar e rejeita colisões. Dados compartilhados são
evidência e catálogo; não são executados e não substituem pesos locais automaticamente.

Para compartilhar pesos no futuro, prefira GitHub Releases ou Git LFS, com hash já
registrado na contribuição, limite de tamanho, revisão e opt-in explícito. Não coloque
checkpoints grandes no Git comum.

Proteja `main` e `knowledge`, exija revisão e atualização fast-forward. Amigos devem
enviar candidatos por forks/pull requests, sem acesso de escrita ao registro campeão.
A comparação CAS local evita sobrescrita no arquivo, mas não substitui CAS remoto:
promoções no GitHub ainda devem usar branch protegida, conferir novamente o HEAD e fazer
push fast-forward, nunca force-push ou rebase automático do registro.

## Validação no GitHub

O workflow `Knowledge integrity / Data only` roda em todo pull request destinado a
`knowledge`. Ele usa o código validador da branch base, nunca instala nem executa código
do pull request e não baixa checkpoints. Somente estes caminhos são aceitos:

- `knowledge/index.json`;
- `knowledge/champion.json`;
- `knowledge/contributions/<sha256>.json`.

Contribuições publicadas são imutáveis. O check também rejeita arquivos órfãos, links,
submódulos, exclusões, documentos acima de 2 MiB, mais de 50 contribuições ou mais de
10 MiB alterados em um único pull request. Forks não podem alterar o campeão; uma
promoção interna precisa avançar exatamente uma geração e apontar para o hash do campeão
anterior.

O rollout tem uma etapa deliberadamente manual: primeiro integre estes arquivos em
`main`, depois abra e revise um pull request de sincronização de `main` para `knowledge`.
Somente após o workflow existir na branch `knowledge`, marque
`Knowledge integrity / Data only` como check obrigatório no ruleset dessa branch. Em
`main`, os checks estáveis são `Code CI / Python` e `Code CI / Bridge contracts`.

Por enquanto o CI público aceita contribuições, índice e promoções internas. O outbox de
candidatos continua local até definirmos um caminho público canônico e sua política de
revisão; não inclua `knowledge/outbox/` nos pull requests.
