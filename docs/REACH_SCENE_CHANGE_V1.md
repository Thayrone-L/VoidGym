# Currículo `reach_scene_change_v1`

Este currículo continua usando exclusivamente pixels RGB e o histórico das ações aplicadas. A política não recebe nome de cena, posição, mapa, rota, telemetria interna nem o sinal `scene_transition` da ponte.

## Detector e recompensa

A baseline visual só é congelada depois de uma janela estável. O detector exige, em ordem:

1. dois frames escuros consecutivos, compatíveis com uma transição visual;
2. retorno a uma imagem informativa, rejeitando telas pretas, brancas e de baixa entropia;
3. novidade suficiente em relação à baseline congelada;
4. três frames consecutivos estáveis no novo regime visual.

O shaping de novidade possui orçamento máximo configurável de `1.0`, mas fica **desativado por padrão (`0.0`)** até ser calibrado. Mesmo quando habilitado, ele só pode ser pago após a confirmação completa de estabilidade. O evento terminal vale `10.0`; o custo por passo é `-0.002`. Término ou truncamento originado no ambiente-base zera todos os componentes positivos. Uma transição operacional inválida lança `InvalidTrainingTransition` antes de chegar ao learner.

O evento terminal é registrado como `visual_regime_change_detected`. Até existir um corpus holdout auditado, ele não deve ser descrito como prova de troca de sala nem como sucesso de rota.

## Execução

O comando abaixo apenas mostra o plano e não abre o jogo:

```powershell
.\.venv\Scripts\python.exe tools\run_scene_change_training.py
```

Uma execução real requer opt-in explícito:

```powershell
.\.venv\Scripts\python.exe tools\run_scene_change_training.py --execute
```

O runner preserva o launcher transacional de saves, bootstrap visual, manifestos schema 2, proveniência com SHA-256 e checkpoints atômicos. Os defaults são 4096 passos, horizonte de 512, rollouts de 256 e checkpoints a cada 1024 passos.

O treino também aceita `--continuous`: nesse modo não há limite total nem truncamento por
horizonte. Rollouts e checkpoints continuam em blocos finitos, o contador do learner não
é reiniciado entre episódios e a retenção é controlada por `--max-checkpoints` (padrão 5).
O probe permanece deliberadamente limitado e não aceita duração contínua.

Tanto o treino quanto o probe aceitam `--stop-file <caminho>`. A existência do
arquivo solicita cancelamento cooperativo: o PPO para pelo callback e o probe
para entre passos/episódios. O manifesto termina em `CANCELLED`, sem checkpoint
parcial, após o teardown normal do ambiente e do processo do jogo.

O manifesto também congela um inventário SHA-256 do save experimental no início
da execução. Como o save pode evoluir entre episódios, eles não são tratados
como réplicas idênticas; avaliações comparáveis exigirão snapshots separados.

## Probe sem learner

Antes de treinar, o detector pode ser exercitado por ações aleatórias reproduzíveis, sem PPO, modelo ou checkpoint:

```powershell
# Apenas exibe o plano
.\.venv\Scripts\python.exe tools\run_scene_change_probe.py

# Opt-in que abre o jogo
.\.venv\Scripts\python.exe tools\run_scene_change_probe.py --execute
```

O probe usa o mesmo isolamento de saves, bootstrap e contrato da tarefa. Seu manifesto próprio registra seed, configurações, hashes de proveniência e resumo por episódio. PNGs são persistidos somente quando há `dark_run`, candidato estável ou evento positivo, sempre acompanhados por SHA-256. Uma execução sem evento positivo termina como `COMPLETED` com `gate_passed=false`; isso é um resultado diagnóstico válido, não falha operacional. Mortes são contabilizadas em `death_episodes`, encerram apenas o episódio e permitem o reset seguinte; ficam separadas de `operational_failures`.
