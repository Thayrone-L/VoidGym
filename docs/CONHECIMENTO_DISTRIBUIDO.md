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
