# Lifecycle de run por Shade

Este contrato separa três níveis: transição, episódio e run lógica. A morte do
Knight encerra o episódio depois da barreira de persistência, mas não encerra a
campanha nem descarta o learner. A primeira morte deixa `recovery_required`; a
recuperação da Shade limpa esse estado. Uma nova morte enquanto a obrigação
estava pendente marca `run_failed=true`, abre uma nova run lógica e mantém a
campanha ativa.

## Fonte dos sinais

O bridge usa APIs do jogo/Modding API 77:

- `BeforePlayerDeadHook` captura o estado **anterior** à morte;
- `AfterPlayerDeadHook` publica `player_dead` e `death_persisted`;
- `SetPlayerBoolHook` e `HeroUpdateHook` acompanham `PlayerData.soulLimited`;
- `BEGIN_EPISODE` reconcilia o estado persistido de `soulLimited` após reload.

O IL de `GameManager.PlayerDead` na versão GOG 1.5.78.11833 confirma que
`AfterPlayerDeadHook` roda somente depois de `orig_PlayerDead`. Essa coroutine
espera o callback de `SaveGame` (`finishedSaving`) antes de chamar
`ReadyForRespawn` e retornar. Portanto `player_dead=true` também é a barreira
explícita que autoriza o supervisor a finalizar/reiniciar o processo. Um frame
com `player_dead=true` e `death_persisted=false` é rejeitado pelo protocolo e
pelo ambiente.

O contrato é restrito ao modo Clássico (`permadeathMode == 0`). `BEGIN_EPISODE`
falha de forma segura nos modos Steel Soul/Godseeker, cujas regras de morte não
são equivalentes.

## Campos de supervisão em `FRAME`

- `player_dead`: terminal do episódio, emitido no hook pós-save;
- `death_persisted`: barreira comprovada para teardown do processo;
- `shade_pending`: `soulLimited` persistido ou obrigação criada pela morte;
- `shade_recovered`: evento latched no episódio quando `soulLimited` passa a falso;
- `run_failed`: a morte começou com Shade pendente.

Esses campos chegam em `info["supervisor"]` e nos logs de transição. Eles não
entram no dicionário de observação (`pixels`, `action_history`) nem alteram
reward. `ShadeRunTracker` mantém o índice da run e a obrigação de recuperação
através de novos episódios e novos processos. O orquestrador deve contabilizar
`run_failed`, reiniciar a run lógica e continuar a campanha com o mesmo learner.

## Limitações

- A detecção depende da semântica de `soulLimited` da versão 1.5.78.11833/API 77.
- A integração foi verificada por inspeção de assembly e testes; não houve teste
  dentro do jogo nesta alteração.
- Alterar a versão do jogo exige revalidar a ordem da coroutine de morte e a
  barreira pós-save antes de instalar a DLL.
