# Process owner e baseline aleatório

## Limite de responsabilidade

`IsolatedEpisodeProcessOwner` é a fronteira entre `HollowKnightEnv` e o processo
do jogo. Ele não escolhe ações, não navega menus e não contém uma rota. Cada
`reset()` corresponde a um novo processo e a uma nova transação de saves.

Fluxo por episódio:

1. `SupervisorEnvironmentBackend` cria um `episode_id` e fornece seu endpoint.
2. O owner encerra, aguarda e restaura qualquer episódio anterior.
3. `start-isolated-hollow-knight.ps1` adquire o lock global de saves, preserva o
   save pessoal e cria a junção para o save experimental.
4. `hk_agent.episode_worker` inicia exatamente um Hollow Knight com as variáveis
   do endpoint do supervisor.
5. No reset seguinte ou no fechamento, o owner cria um pedido de parada. O
   worker fecha somente o PID que criou; o launcher restaura o save pessoal.

O owner nunca mata processos pelo nome. Se o encerramento exceder o limite, ele
mantém a transação como ativa e falha de forma fechada, para que uma nova chamada
ou a recuperação explícita possa concluir a restauração. O lock do launcher e a
barreira dentro do owner impedem episódios concorrentes.

Um reset que excede o timeout cancela sua geração de lançamento. A verificação é
feita imediatamente antes da criação do processo; se o timeout competir com uma
factory já iniciada, o owner assume o processo retornado e solicita teardown e
restauração transacional. O cancelamento não espera uma factory bloqueada.

Os guards do launcher e da recuperação comparam o caminho executável canônico,
não apenas o nome do processo. Um homônimo com caminho verificável diferente não
bloqueia; um processo homônimo cujo caminho não possa ser verificado bloqueia de
forma segura.

## Menu bootstrap

O bootstrap de menu é infraestrutura externa à política. Este marco não contém
nem executa uma sequência de menu. Antes de uma execução real, um coordenador
separado deverá colocar o jogo em um estado observável de gameplay sem registrar
essas ações como experiência da política.

## Baseline aleatório

`tools/run_random_baseline.py` executa uma política mínima que:

- semeia e amostra somente `env.action_space`;
- tem orçamento curto e explícito;
- exige recompensa constante igual a zero;
- registra ações, contagens, término e falhas em um manifesto atômico;
- rejeita qualquer transição cujo `info.valid` não seja exatamente `True`;
- sempre fecha o ambiente ao final.

O comando é dry-run por padrão e não inicia o jogo:

```powershell
.\.venv\Scripts\python.exe .\tools\run_random_baseline.py --seed 7 --steps 32
```

A opção `--execute` é a autorização explícita para uma execução real. Ela ainda
pressupõe que o bootstrap externo de menu esteja resolvido; não deve ser usada
para tentar ensinar navegação de menus à política de gameplay.

## Artefatos

- `artifacts/baseline/baseline-manifest.json`: resultado do baseline.
- `artifacts/baseline/experience/`: transições válidas e operações rejeitadas.
- `artifacts/baseline/episodes/<episode_id>/lifecycle.json`: ciclo do owner.
- `worker-result.json`: resultado de encerramento do processo do jogo.
