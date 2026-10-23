# Prompt 85 — política universal e pré-treino v2

**Estado:** implementação e pré-treino concluídos; a avaliação completa held-out está em execução após correção de uma semente inválida do nível 7.

## Modelo e treinamento

A política universal combina um encoder visual IMPALA, histórico de ações, LSTM recorrente, ator universal com máscara e crítico. O encoder alimenta também uma cabeça semântica e cabeças auxiliares de dinâmica inversa e movimento do herói. O checkpoint é salvo em partes e inclui a tabela de ações, o currículo e metadados de treino.

- Pré-treino final: `runtime/v2/pretrain/pretrain-resume-2026-10-23`.
- Checkpoint final: `universal-000697088`, com **697.088 passos**; treinamento retomado a partir de `universal-000525824`.
- Execução terminou normalmente em 23/10/2026 às 16:26, após aproximadamente 8 h. Throughput final: **14,62 passos/s**.
- Perfil usado: CPU, 3 ambientes, 3 threads; GPU não foi selecionada (`train_device=cpu`), VRAM usada 0. O caminho CUDA não foi exercitado nesta máquina nesta execução.
- Último registro: `L_ppo=-0,0177`, `L_semantic=0,3728`, `L_inverse_dynamics=0,8060`, `L_hero_motion=0,3602`, entropia `1,4150`, KL `0,0132`, explained variance `0,9427`, nível de currículo 1 e sucesso na janela de 100 episódios `69%`.
- O alvo configurado de 20 milhões de passos não foi atingido; a execução terminou pelo limite de 8 horas. O checkpoint final íntegro está disponível com seus módulos e `manifest.json`.

Curvas: `85_curvas.png`.

## Avaliação

A primeira tentativa avaliou os níveis 0–6 em modo determinístico e chegou à campanha 24 do nível 7 antes de encontrar uma composição sem rota solucionável (seed `155230`). Os GIFs de níveis 3, 4 e 6 foram salvos. A causa era a ausência de nova tentativa para a composição conectada, embora cada sala individual tivesse prova válida. `generate_room_set` agora tenta até 20 composições; o seed que falhou passa na prova e há teste de regressão.

A nova avaliação usa split `test`, 100 episódios por nível 0–6 e 30 campanhas no nível 7, modos determinístico e estocástico, limite de 1.500 passos por episódio e até 5.000 quadros para percepção. Ela salva o progresso após cada par modo/nível em `progress_eval.json` para permitir retomada após interrupções. A tabela final e a matriz serão acrescentadas aqui quando a execução terminar.

### Inspeção visual preliminar dos GIFs

- Nível 3: o agente avança até a parede/obstáculo elevado e parece ficar preso ali, sem demonstrar salto consistente para superar a estrutura.
- Nível 4: o personagem permanece no fundo do buraco e não completa a subida pela borda.
- Nível 6: aproxima-se da parede quebrável, mas não há evidência visual de que ataque e atravesse o bloqueio.
- Nos três GIFs, o progresso é pequeno ou ausente; não há evidência de desvio consistente de perigos.
- Essas gravações vêm da primeira avaliação interrompida no nível 7 e serão substituídas/confirmadas pela execução retomável.

## Validação

- Suíte completa: **783 passaram, 3 ignorados, 3 avisos e 8 subtestes passaram** (`242,37 s`). Os avisos remanescentes são conversões de tensores do Stable-Baselines3.
- `ruff check src tools tests`: passou.
- `mypy src`: passou (98 arquivos).
- `git diff --check`: passou.
- Smoke de treino recorrente com ambientes subprocessados e currículo: concluído previamente até 25.088 passos, incluindo resets entre níveis.
- Amostragem de experiência a 1% e carregamento do checkpoint universal foram verificados previamente.

## Limitações e próximos passos

A campanha ficou aquém de 20M passos e avançou apenas até o nível de currículo 1; portanto, não há evidência suficiente para afirmar que a política passa com folga nos níveis 0–6. O modelo foi treinado em CPU apesar da RTX 3060 prevista pelo perfil de hardware, e a avaliação CUDA/AMP ainda não foi feita. A decisão de promover o checkpoint para os prompts 86/87 deve considerar os resultados held-out abaixo e essas limitações.

## Arquitetura e hiperparâmetros

O encoder recebe `uint8 (B,12,108,192)` (quatro quadros RGB), aplica blocos IMPALA de 16/32/32 canais e produz mapa `(B,32,14,24)` e vetor ReLU de 256. A cabeça semântica emite logits `(B,10,27,48)`. Vetor visual (256), embedding do histórico de ações (64) e extras do jogo (256; zerados no ginásio) entram na LSTM de 256 unidades. A cabeça `explore` gera logits mascarados sobre as ações universais; o crítico recebe a LSTM e, separadamente, um vetor privilegiado de 64 dimensões. As cabeças auxiliares predizem dinâmica inversa e movimento do herói.

Configuração final registrada em `runtime/v2/pretrain/pretrain-resume-2026-10-23/config.json`: PPO recorrente, rollout 256, batch 512, 4 épocas, LR `2,5e-4`, gamma `0,995`, GAE `0,95`, entropia `0,01`, clip `0,2`, sequência 32; coeficientes semântico/inverso/movimento `0,5/0,1/0,1`, com aumento de deslocamento aleatório. Foram usados 3 ambientes e 3 threads de CPU. O ciclo foi limitado a 8 horas.
