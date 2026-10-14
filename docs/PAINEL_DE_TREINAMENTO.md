# Painel local de treinamento

O Void Lab é uma interface web local para acompanhar manifests, versionar campanhas e
acionar os runners já auditados. Ele usa somente a biblioteca padrão do Python e escuta
exclusivamente em `127.0.0.1`.

## Iniciar

Modo recomendado para inspeção e testes de configuração, sem abrir o jogo:

```powershell
.\.venv\Scripts\python.exe tools\serve_training_dashboard.py
```

Abra `http://127.0.0.1:8765`. Nesse modo, qualquer campanha lançada é um dry-run do runner.

Para permitir que o botão de execução real acrescente `--execute` ao runner:

```powershell
.\.venv\Scripts\python.exe tools\serve_training_dashboard.py --allow-execute
```

Mesmo nessa modalidade, a interface exige que a pessoa marque explicitamente “Executar no
jogo” para cada lançamento. O servidor não é publicado na rede.

## Lançar pelo navegador

1. Preencha a campanha, escolha `Usar save existente` ou `Iniciar novo jogo` e o slot.
   Em **Duração**, use `Limitada por passos` para uma rodada finita ou `Contínua até
   solicitar parada` para aprendizado sem limite artificial de passos.
2. Clique em **Salvar versão**. A campanha passa a aparecer em **Campanha salva**.
3. Marque **Executar no jogo (não dry-run)**.
4. Clique em **Lançar** e acompanhe `Processo ativo`, o log e o histórico de runs.

Para encerrar uma campanha contínua, use **Solicitar parada** e aguarde `CANCELLED` ou
outro estado terminal, garantindo que a restauração transacional dos saves termine.
Se o painel reiniciar, ele reanexa o registro do runner vivo, continua bloqueando novos
lançamentos e mantém disponível a parada cooperativa por `stop.request`.

## Segurança e persistência

- Apenas `run_first_training.py`, `run_scene_change_probe.py` e
  `run_scene_change_training.py` estão na allowlist.
- Os processos usam uma lista fixa de argumentos e `shell=False`; texto arbitrário nunca é
  convertido em comando.
- Existe somente um processo de campanha ativo por servidor.
- No modo contínuo não há um inteiro enorme usado como falso infinito e o horizonte não
  trunca episódios. Os rollouts do PPO continuam finitos internamente; morte e outros
  eventos de gameplay encerram um episódio, não a campanha. A campanha só encerra por
  parada cooperativa ou falha operacional.
- **Checkpoints retidos** limita o uso de disco (padrão: 5). A rotação acontece somente
  depois que o novo arquivo atômico foi salvo e teve o SHA-256 calculado. O manifesto
  mantém `checkpoints` para os arquivos retidos e `pruned_checkpoints` para o histórico
  dos removidos; nunca se remove o único checkpoint válido.
- Os pesos **Inimigo derrotado**, **Teto por mortes**, **Recuperar a Sombra**,
  **Dano / HP perdido**, **Geo ganho** e **Geo gasto** começam em `2`, `6`, `2`,
  `-0,25`, `0,01` e `0,02`. A morte de inimigo e a recuperação da Sombra são
  eventos confirmados pelo bridge; a recuperação só é paga uma vez por episódio.
  Eles só consomem deltas explícitos confirmados
  (`enemy_kills_since_last_frame`, `hp_lost_since_last_frame`,
  `geo_gained_since_last_frame` e `geo_spent_since_last_frame`); não entram na observação
  da política e nenhuma derrota de inimigo é inferida visualmente pelo painel. Os deltas de
  Geo são causais: coleta é confirmada no gatilho físico `GeoControl.OnTriggerEnter2D` e compra por
  `HeroController.TakeGeo`; restauração da Shade usa `AddGeoQuietly` e morte altera o saldo
  diretamente, portanto não viram falsos ganho/gasto. O valor contabilizado é ainda o delta
  real do saldo depois da chamada, limitado pelo que o jogo aceitou. No currículo de cenário,
  inimigos e Geo compartilham um orçamento positivo de 25% da recompensa de progresso; no
  currículo motor, Geo tem teto total de 2,5 por episódio.
- O espaço da política contém ações temporais explícitas: pulo curto (`50 ms`), pulo
  longo (pelo menos `120 ms`, limitado a `250 ms`), movimento+pulo, pulo+ferrão e
  movimento+pulo+ferrão. São combinações simultâneas, não sequências roteirizadas.
  O histórico registra o identificador da ação temporal, portanto pulo curto e longo
  continuam distinguíveis mesmo usando os mesmos botões brutos.
- A primeira morte encerra o episódio e deixa recuperação de Shade pendente. Recuperar a
  Shade limpa a pendência. Morrer novamente com a pendência ativa registra uma run lógica
  falha e inicia outra, mas preserva campanha, PPO, otimizador e timesteps.
- O PPO de mudança de cenário permanece bloqueado até um probe positivo ser revisado
  manualmente. A aprovação referencia o manifest do probe por SHA-256; um evento visual
  detectado automaticamente, sozinho, nunca abre o gate.
- Requisições mutáveis precisam do token efêmero da sessão e rejeitam origens externas.
- “Parar” cria atomicamente um arquivo `stop.request`. O runner observa o pedido entre
  rollouts completos do PPO, salva um checkpoint final retomável e executa o
  teardown normal do ambiente, processo e isolamento de saves. O painel mostra
  `STOP_REQUESTED` enquanto aguarda e `CANCELLED` após o encerramento; ele não envia sinais,
  `terminate` ou `kill` ao processo da campanha.
- Cada alteração de campanha cria uma nova versão imutável em
  `artifacts/control-plane/campaigns/<id>/vNNNN.json`.
- A origem também faz parte da versão: `existing_save` exige um slot experimental existente
  entre 1 e 4; `new_game` exige que o slot esteja vazio e nunca sobrescreve um save.
- O runner resolve a navegação do menu para o slot escolhido, registra os passos e o hash do
  plano resolvido na proveniência da run. Campanhas antigas continuam como save existente,
  slot 1.
- Logs e registros dos launches ficam em `artifacts/control-plane/launches/`.
- Ao iniciar, o painel reconcilia `RUNNING`/`STOP_REQUESTED`: um PID ainda vivo é marcado
  `reattached`; um processo já ausente recebe estado terminal explícito, pois seu código
  de saída não pode ser recuperado após o reinício.

Os runners de PPO aceitam `--resume-checkpoint <arquivo.zip>`. A retomada usa `PPO.load`
com o ambiente atual e só é aceita quando SHA-256, contrato da tarefa e proveniência
coincidem com o manifesto de origem; não basta apontar para qualquer arquivo `.zip`.

O painel lê os manifests sob `artifacts/training` e `artifacts/probes`. Ele não altera nem
interpreta arquivos de save.
