# Primeira rodada de treinamento

## Resultado

Em 12 de outubro de 2026 foi concluída a primeira rodada real de PPO contra o
Hollow Knight isolado. Esta rodada é um diagnóstico de aprendizado motor, não
uma alegação de progresso, conclusão de sala ou habilidade de speedrun.

- perfil do jogo: `gog-1.5.78.11833`;
- algoritmo: PPO, CNN pequena iniciada aleatoriamente;
- seed: `7`;
- observação da política: pixels reduzidos para `160x90`, quatro frames, mais
  histórico das próprias ações;
- ações motoras iniciais: `NOOP`, `LEFT`, `RIGHT`, `JUMP` e `ATTACK`;
- o currículo de mudança de cenário também expõe golpes direcionais e golpes
  combinados com salto/movimento, sem macros pré-programadas;
- recompensa: `visual_motion_v1`, calculada somente de pixels permitidos;
- orçamento: 256 transições válidas em dois rollouts de 128;
- recompensa total registrada: `26.742937144951487`;
- checkpoints: passos 128 e 256, ambos recarregados e verificados;
- status final: `COMPLETED`.

Artefatos da execução:

`artifacts/training/visual-motion-v1/run-20261012T181113/`

## Calibração

Antes de treinar, uma sessão separada mediu cinco amostras `NOOP` e cinco
amostras com ações horizontais alternadas. O ruído visual `NOOP` ficou entre
aproximadamente 2,5% e 3,6%; as ações ativas ficaram entre 17% e 74%. O limiar
resolvido foi `0.05371062992125984`.

Se o movimento ativo não superar o ruído por margem suficiente, o treinamento
falha antes de criar o modelo.

## Segurança e separação de responsabilidades

- O bootstrap de menu ocorre antes de `BEGIN_EPISODE` e nunca entra na
  experiência da política.
- Um verificador visual versionado precisa confirmar gameplay; capturas apenas
  diferentes não bastam, pois menus também são animados.
- Saves pessoais são preservados pelo launcher transacional. Os quatro
  manifests de processo da rodada terminaram como `COMPLETED`.
- A pasta pessoal foi restaurada e não permaneceu como junction.
- As 256 transições foram válidas; nenhuma transição operacional inválida foi
  consumida pelo PPO.
- `visual_motion_v1` pode ser explorada por movimento repetitivo. Por isso o
  manifesto traz `diagnostic_training=true` e não chama o sinal de sucesso no
  jogo.

## Como repetir

Dry-run, sem abrir o jogo:

```powershell
.\.venv\Scripts\python.exe .\tools\run_first_training.py
```

Execução real curta:

```powershell
.\.venv\Scripts\python.exe .\tools\run_first_training.py --execute --timesteps 256 --checkpoint-interval 128
```

O runner aceita `--stop-file <caminho>`. Quando esse arquivo passa a existir, o
callback do PPO encerra a coleta cooperativamente, grava `CANCELLED`, não produz
checkpoint do rollout interrompido e executa o fechamento normal do ambiente.

Com `--continuous`, `total_timesteps` deixa de ser um limite e o horizonte da tarefa é
desativado. O modelo continua entre episódios com `reset_num_timesteps=False` e grava
checkpoints a cada `--checkpoint-interval`; `--max-checkpoints` (padrão 5) aplica rotação
segura. Esse modo exige `--stop-file` quando lançado pelo painel e só termina por esse
pedido ou por falha operacional. O callback já reconhece um futuro `info.run_failed=true`
do ambiente como falha da campanha; a detecção concreta de Shade não é inferida aqui.

O próximo marco é `reach_scene_change_v1`: um evento booleano versionado de
mudança estável de cena, sem expor nome, posição, direção ou rota. Depois disso,
o checkpoint deve ganhar avaliação congelada e estado de RNG para retomada
reprodutível.
