# Plano mestre

## 1. Resultado pretendido

Construir um sistema local no qual uma política, iniciada sem conhecimento específico de Hollow Knight, aprenda por interação visual e mantenha conhecimento derivado das próprias experiências. A primeira meta de produto é exploração; conclusão, conquistas, sobrevivência e velocidade são extensões posteriores e isoladas.

O primeiro produto técnico não é uma IA treinada. É um ambiente confiável que repete o ciclo:

```text
observação válida -> ação limitada -> execução medida -> nova observação
       ^                                             |
       +-------------- reinício auditável -----------+
```

## 2. Invariantes

Estas regras valem em todas as fases:

- Nenhuma posição interna, nome de sala, mapa completo, estado oculto de inimigo, rota ou estratégia entra na observação da política.
- `info` e logs de diagnóstico não entram automaticamente no modelo, na memória ou no cálculo de novidade.
- O crítico, se houver arquitetura ator-crítico, recebe a mesma fronteira de informação da política.
- Toda recompensa é soma de componentes nomeados, registrados e versionados.
- Cada experiência aceita liga exatamente um comando a observações anterior e posterior.
- Frames repetidos, fora de ordem, incompletos ou obtidos durante transição inválida não viram experiência de treinamento.
- Uma desconexão não pode deixar botão pressionado.
- Um save pessoal não pode ser selecionado por configuração experimental.
- Avaliação usa pesos congelados, seeds/configurações registradas e todas as tentativas são reportadas.
- Paralelismo só é ativado depois de provar isolamento por processo.

## 3. Não objetivos iniciais

- Zerar ou platinar o jogo em prazo prometido.
- Determinismo bit a bit.
- Acelerar a simulação.
- Controlar menus gerais como parte da ação aprendida.
- Fazer o agente aprender com guias, vídeos, mapas, wikis ou modelos que possam memorizar o jogo.
- Resolver exploração de longo prazo antes de validar uma tarefa motora curta.
- Otimizar duas a quatro instâncias antes de medir uma.

## 3.1 Perfis de versão

A versão deixa de ser uma restrição global. Cada execução declara `game_profile`, e nenhum checkpoint ou save atravessa perfis silenciosamente.

| Perfil | Versão proposta | Uso | Regra |
|---|---|---|---|
| `development` | `1.5.78.11833`, condicionada ao diagnóstico | ponte, captura, input, aprendizado inicial e exploração | preferir compatibilidade e ferramentas estáveis |
| `speedrun_any_nmg_1221` | `1.2.2.1` | etapa futura de velocidade na categoria correspondente | regras e cronômetro congelados no manifesto |
| `speedrun_1432_plus` | `1.4.3.2` ou posterior aceito | categorias modernas/NMG correspondentes | versão exata registrada; não agrupar builds diferentes nos dados |
| `current` | versão instalada atual | compatibilidade e comparação, se útil | nunca presumir que a API funciona |

`1.2.2.1` é uma versão antiga muito usada em Any%/NMG, mas não deve ser chamada de primeira versão do jogo. A versão correta para velocidade depende da categoria. As tabelas atuais do Speedrun.com exibem runs Any% em `1.2.2.1` e categorias separadas `1.4.3.2+`; isso será reconfirmado quando ativarmos velocidade.

Manter múltiplas versões exige instalações autônomas, mods, saves, configurações e diretórios de execução separados. O projeto não depende de Steam. Saves nunca serão compartilhados entre versões sem uma migração testada. Transferir pesos entre perfis é um experimento explícito de mudança de domínio; o padrão é bloquear incompatibilidade e exigir decisão. Como Unity pode manter o mesmo `Application.persistentDataPath` para cópias diferentes, a separação de executáveis não será aceita como evidência de isolamento dos saves.

## 4. Arquitetura de referência

```text
                         PROCESSO DO JOGO (instância N)
  +------------------------------------------------------------------+
  | Hollow Knight + Modding API + ponte C#                           |
  |                                                                  |
  | captura -> carimbo/seq -> transporte      aplicador de entradas  |
  |    |                 |                         ^                  |
  | eventos permitidos  watchdog -> RELEASE_ALL -+                   |
  +----------------------+---------------------------+---------------+
                         | loopback autenticado      |
                         v                           |
  +------------------------------------------------------------------+
  | Supervisor Python                                                |
  | sessão | protocolo | validação | Gymnasium Env | logs | reset    |
  +---------------------------+--------------------------------------+
                              |
             +----------------+----------------+
             |                                 |
             v                                 v
  armazenamento por execução        política/treinador (fase posterior)
  manifestos, JSONL, amostras        uma cópia, inferência em lote
  de frames, métricas, vídeo         e estado por ambiente
```

### 4.1 Ponte C#

Responsabilidades permitidas:

- Capturar a visão normal do jogo, com HUD, após a renderização apropriada.
- Aplicar e liberar entradas por instância.
- Emitir heartbeat, versão, capacidades e eventos operacionais.
- Detectar eventos explicitamente autorizados, como dano do jogador, morte, carregamento e conclusão de reinício.
- Recusar comandos vencidos, duplicados, fora de ordem ou de outra sessão.
- Liberar entradas em timeout, exceção, desconexão, troca de cena inválida ou encerramento.

Responsabilidades proibidas:

- Tomar decisões de navegação ou combate.
- Expor estado interno não autorizado.
- Calcular novidade usando nomes de cena ou coordenadas.
- Restaurar saves enquanto o jogo ainda os pode estar usando.

### 4.2 Supervisor Python

- Inicia/encerra processos e associa PID, instância, diretório e sessão.
- Faz o handshake e valida capacidades.
- Implementa `reset`, `step` e `close`.
- Separa `terminated` de `truncated`.
- Garante limites de tempo e armazenamento.
- Persiste manifesto e logs antes de habilitar treinamento.
- Disponibiliza parada de emergência fora da política.

### 4.3 Transporte

Escolha inicial: TCP binário em `127.0.0.1`, com enquadramento de mensagens, token aleatório por execução e conexão iniciada pela ponte em direção ao supervisor.

Motivos:

- interoperabilidade simples entre .NET Framework e Python;
- uma conexão por instância sem depender de foco;
- fácil correlação, timeout e captura de diagnóstico;
- nenhuma interface exposta à rede se o bind permanecer em loopback.

Fase 0 usa frames comprimidos ou brutos de baixa frequência para privilegiar correção. Se a cópia dominar a latência, o plano de otimização é controle por TCP e frames por memória compartilhada com slots versionados. Essa otimização exige benchmark; não é requisito do primeiro pacote.

### 4.4 Captura

Ordem de implementação:

1. Captura correta ao fim do frame, mesmo que síncrona e lenta.
2. Verificação de que HUD, transições, pausa e resolução estão corretos.
3. `AsyncGPUReadback` quando suportado, mantendo fallback.
4. Redução de resolução e formato no pipeline, nunca escondida dentro da política.

Configuração diagnóstica inicial proposta: `320x180 RGB`, 10–15 observações/s, sem compromisso de treino. Candidato de treino: `160x90`, RGB ou luminância decidida por ablação, 4 frames empilhados. Esses valores só se tornam padrão depois dos testes de legibilidade e latência.

### 4.5 Controle interno

A ponte terá uma interface de backend. O spike de viabilidade testa, nessa ordem:

1. injeção no estado de entrada que o jogo realmente consome;
2. dispositivo virtual/estado da Unity Input System, se compatível;
3. hook mais específico e documentado como último recurso.

Não será usado `SendInput`, automação de janela ou alternância de foco como mecanismo principal. O backend aprovado precisa controlar duas janelas simultaneamente em teste posterior, cada uma com comandos opostos e sem interferência cruzada.

## 5. Fronteiras de informação

Há quatro classes independentes:

| Classe | Exemplos | Política |
|---|---|---|
| Observação do agente | imagem, histórico de ações, objetivo | pode entrar no modelo |
| Sinal de tarefa | morte, dano próprio, sucesso declarado | só entra pela recompensa se documentado |
| Operação | heartbeat, frame inválido, PID, carregamento | apenas controle e `info` |
| Diagnóstico privilegiado | posição, cena, detalhes de save | desativado por padrão; nunca no modelo |

Um dado estar em `info` não autoriza seu uso. Consumidores de treino receberão uma visão filtrada por allowlist, não o dicionário bruto.

## 6. Estado operacional

Cada instância segue a máquina:

```text
CREATED -> CONNECTING -> HANDSHAKING -> READY
READY -> RESETTING -> READY
READY -> STEPPING -> READY
qualquer estado -> FAULTED -> STOPPING -> STOPPED
```

Regras:

- `step` só é aceito em `READY`.
- Só existe um comando em voo por instância na versão 1.
- Timeout move a instância para `FAULTED`, executa `RELEASE_ALL` e produz truncamento operacional.
- Mensagem com `session_id`, `episode_id` ou sequência incorretos é rejeitada e logada.
- Depois de `terminated` ou `truncated`, um novo `step` é proibido até `reset`.

## 7. Ações

### 7.1 Espaço bruto inicial

O agente controla um vetor de botões, não macros semânticas:

- direções: esquerda, direita, cima, baixo;
- ações de jogo: pular, atacar, dash, foco/conjuro, super dash, dream nail.

Os botões existem no protocolo mesmo antes de uma habilidade estar disponível; o ambiente não envia uma máscara secreta baseada no save. Pressionar uma ação indisponível apenas não produz efeito observável. Pause, confirmar, cancelar, menus e saída pertencem ao canal do supervisor.

### 7.2 Segurança e normalização

- esquerda+direita e cima+baixo são normalizados por regra configurada e registrada;
- toda ação tem duração máxima;
- `RELEASE_ALL` tem prioridade sobre qualquer ação;
- ataques ou saltos compostos só entram como opções aprendidas depois de serem adquiridos por treino, não codificados como estratégia;
- ação solicitada e ação aplicada são ambas registradas.

### 7.3 Frequência

Não presumir 60 decisões/s. Medir o intervalo real e escolher `action_repeat`/duração de forma que:

- toque e retenção sejam distinguíveis;
- nenhuma entrada fique ativa além do lease;
- a taxa seja sustentável sem filas crescentes;
- animações e transições não sejam amostradas de modo enganoso.

## 8. Observação

Observação candidata na primeira tarefa aprendida:

```text
pixels uint8 [K, H, W, C]
ações aplicadas nas K-1 transições anteriores
objetivo da tarefa codificado sem conhecimento do mapa
```

Antes do treino serão testados:

- legibilidade do Knight, plataformas, projéteis e HUD;
- diferença entre toque e retenção;
- latência ação-imagem;
- efeito de RGB versus luminância;
- 2, 4 e 8 frames empilhados;
- recorte versus letterbox sem deformar proporções.

Não incluir áudio inicialmente. Ele pode virar uma observação futura somente após contrato separado.

## 9. Reinício e saves

### 9.1 Tipos de reinício

- `soft_reset`: volta à condição da tarefa sem reiniciar processo, se comprovadamente confiável.
- `episode_reset`: encerra a tentativa e restaura um save experimental de referência.
- `process_reset`: encerra o jogo, restaura artefatos e inicia uma nova sessão.
- `campaign_reset`: nova campanha para avaliação de conhecimento persistente.

### 9.2 Política de proteção

Antes de qualquer escrita automatizada:

1. identificar plataforma, versão, executável e todos os locais alterados ao salvar;
2. calcular inventário/hash dos saves pessoais;
3. criar backup versionado fora da pasta ativa;
4. desabilitar ou controlar sincronização em nuvem durante testes destrutivos;
5. provar a leitura e escrita em um save experimental;
6. provar que um crash no meio do reset não toca nos saves pessoais;
7. bloquear execução se a allowlist de caminhos não for satisfeita.

Escolher outro slot não é isolamento suficiente. Para paralelismo, cada processo precisa de virtualização de caminho de save verificada. Se a API atual não permitir isso, paralelismo fica bloqueado até existir hook seguro ou isolamento por conta/ambiente do Windows.

## 10. Recompensa

### 10.1 Fases 0–2

Recompensa é sempre zero. O sistema coleta apenas evidência operacional.

### 10.2 Primeira tarefa curta

Componentes candidatos, cada um versionado:

- `task_success`: único positivo terminal;
- `player_damage`: custo pequeno, apenas se o exercício exigir;
- `player_death`: custo terminal;
- `time_cost`: custo mínimo para evitar espera infinita;
- `invalid_transition`: nunca recompensa; a transição é descartada e a tentativa truncada.

Não usar posição interna para progresso. A condição de sucesso deve ser visualmente definível ou um evento de tarefa explicitamente autorizado e sem revelar caminho.

### 10.3 Exploração

Só entra depois da tarefa curta. Candidatos serão comparados:

- Random Network Distillation com redes iniciadas aleatoriamente;
- contagem aproximada em embeddings treinados apenas nas experiências;
- memória episódica de similaridade visual;
- grafo de lugares construído por observação e transições vividas.

Testes adversariais obrigatórios: partículas, HUD piscando, atacar o vazio, alternar direção, ficar parado, entrar/sair da mesma tela e morrer para restaurar um estado. Qualquer estratégia que gere novidade crescente nesses casos reprova.

## 11. Estratégia de aprendizado

### 11.1 Baselines antes de RL

1. política aleatória com a mesma distribuição de duração;
2. política de segurança/controle simples, sem conhecimento da solução;
3. política aprendida.

O baseline simples serve para medir se o ambiente é aprendível e se a recompensa está correta; não vira demonstração do agente.

### 11.2 Primeiro candidato

CNN pequena iniciada aleatoriamente + PPO + frames empilhados. PPO é candidato, não decisão irrevogável. O conjunto final depende de:

- espaço de ação escolhido;
- velocidade de coleta;
- estabilidade dos resets;
- memória exigida pela tarefa;
- orçamento de VRAM compartilhado com o jogo.

Critério para recorrência: tarefas com desempenho significativamente limitado por aliasing temporal após ablação de 2/4/8 frames. LSTM/GRU não entra apenas por parecer mais sofisticado.

### 11.3 Checkpoint completo

Salvar atomicamente:

- pesos e otimizador;
- normalizadores;
- configuração resolvida;
- contadores globais e por ambiente;
- estados aleatórios de Python, NumPy e PyTorch;
- versão de código e dependências;
- esquema do protocolo/recompensa;
- memória persistente, quando existir;
- manifesto dos saves/ambiente.

Um checkpoint incompatível deve falhar com mensagem clara, não ser carregado parcialmente.

## 12. Fases e portões

### Fase 0 — inventário reproduzível

Entregáveis:

- relatório de SO, CPU, RAM, GPU, driver, Python, .NET SDK/runtime;
- plataforma, caminho, hash/versão do executável e DLLs relevantes;
- inventário de saves e sincronização;
- seleção e materialização do perfil `development`, com Modding API compatível.

Portão: nenhuma alteração até backup e inventário existirem.

### Fase 1 — ponte mínima

Entregáveis:

- mod carrega e anuncia versão/capacidades;
- conexão loopback autenticada;
- heartbeat, `PING`, `RELEASE_ALL` e encerramento limpo;
- nenhum evento de gameplay ainda necessário.

Portão: 100 ciclos conectar/desconectar sem entrada presa ou crash não explicado.

### Fase 2 — observação, ação, reset e logs

Entregáveis:

- captura correlacionada;
- ações brutas com lease;
- reinício experimental;
- Gymnasium Env validável;
- logs e amostras de frames;
- parada de emergência.

Aceite:

- 20 resets consecutivos sem intervenção;
- 30 minutos de execução sem dessincronização silenciosa;
- cada transição válida correlacionada;
- saves pessoais com hashes inalterados;
- controle em segundo plano comprovado;
- kill do Python e queda de conexão liberam todos os botões.

### Fase 3 — habilidades motoras

- definir tarefa curta somente após ver o ambiente;
- calibrar ação/frequência/observação;
- executar baselines;
- treinar rede pequena;
- avaliar pesos congelados em tentativas separadas.

Portão: intervalo de confiança ou evidência repetida de melhora, sem seleção de melhores episódios.

### Fase 4 — exploração e memória

- introduzir novidade em experimento isolado;
- reconhecer lugares por experiência visual;
- construir grafo de transições vividas;
- testar ciclos e farming de novidade;
- persistir memória entre tentativas sem misturar avaliação.

### Fase 5 — paralelismo

- provar save, entrada, observação, logs e reset independentes;
- medir 1, 2 e 4 instâncias com mesma configuração;
- compartilhar uma política e fazer inferência em lote;
- escolher pela aprendizagem por hora e estabilidade, não só FPS.

### Fase 6 — objetivos avançados

Ordem recomendada: conclusão, sobrevivência, conquistas e velocidade. Cada objetivo ganha recompensa, protocolo de avaliação e memória separados. Velocidade também ganha um perfil de versão/categoria imutável; resultados de versões distintas não são comparados como se viessem do mesmo ambiente.

## 13. Avaliação

Para cada experimento registrar antecipadamente:

- hipótese;
- métrica primária;
- orçamento de passos/tempo;
- conjunto de seeds/condições;
- critério de parada;
- baseline;
- número de tentativas;
- regras para falhas operacionais;
- versão de recompensa e observação.

Resultados mínimos:

- média, mediana, dispersão e intervalo de confiança;
- taxa de sucesso;
- mortes e duração;
- distribuição, não apenas melhor episódio;
- tempo de coleta e atualização;
- transições válidas e descartadas;
- RAM/VRAM/CPU e falhas por hora.

Vídeos são evidência qualitativa, não substituem métricas.

## 14. Organização futura do código

```text
bridge/                 mod C# e testes
controller/             pacote Python
  hk_env/               protocolo, processo, Env e wrappers
  training/             algoritmos e configuração
  evaluation/           runners e relatórios
configs/                arquivos declarativos versionados
schemas/                protocolo, logs e manifestos
tests/                  unitários, integração e fault injection
tools/                  diagnóstico e manutenção
runs/                   ignorado pelo Git; dados por execução
docs/                   decisões e procedimentos
```

Dados de execução nunca são misturados com código. Cada `run_id` contém manifesto imutável, configuração resolvida e subdiretórios por instância/episódio.

## 15. Fontes técnicas verificadas em 2026-10-09

- [Repositório da Hollow Knight Modding API](https://github.com/hk-modding/api)
- [Issue de compatibilidade com o jogo 1.5.12459](https://github.com/hk-modding/api/issues/179)
- [Lumafly](https://github.com/TheMulhima/Lumafly)
- [Documentação de modding para 1.5.x](https://github.com/PrashantMohta/ModdingDocs)
- [Local oficial de saves no Windows](https://www.hollowknight.com/help)
- [Gymnasium Env API](https://gymnasium.farama.org/api/env/)
- [Stable Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)
- [Unity 2020.2 AsyncGPUReadback](https://docs.unity3d.com/2020.2/Documentation/ScriptReference/Rendering.AsyncGPUReadback.html)
- [PyTorch: instalação local](https://pytorch.org/get-started/locally/)
- [Hollow Knight no Speedrun.com](https://www.speedrun.com/hollowknight)
- [Guia comunitário de múltiplos patches](https://www.speedrun.com/hollowknight/guides/jlten)

As fontes sobre ferramentas servem à engenharia. Nenhum conteúdo específico do mundo do jogo será transformado em observação, memória ou dado de treino.
