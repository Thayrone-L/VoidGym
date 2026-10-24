# Prompt 85 — política universal e pré-treino v2

**Estado:** implementação, pré-treino e avaliação held-out completos. A avaliação documenta metas não alcançadas; o checkpoint ainda não deve ser considerado pronto para promoção sem decisão explícita sobre essas limitações.

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

Avaliação completa no split `test`: 100 episódios por nível 0–6 e 30 campanhas no nível 7, em modos determinístico e estocástico, limite de 1.500 passos. A tabela completa está em `85_eval/tabela.md`.

| Nível | Sucesso determinístico | Sucesso estocástico |
|---:|---:|---:|
| 0 | 76% | 81% |
| 1 | 69% | 78% |
| 2 | 62% | 62% |
| 3 | 10% | 46% |
| 4 | 1% | 45% |
| 5 | 74% | 69% |
| 6 | 18% | 72% |
| 7 | 0% | 3,3% |

Percepção semântica foi reavaliada com 5.000 quadros de teste, 625 de cada nível, em vez de concentrar a amostra no nível 0. IoU: EMPTY 0,893; SOLID 0,377; PLATFORM 0,168; HAZARD 0,000; ENEMY 0,000; PLAYER 0,297; BREAKABLE 0,000; PICKUP 0,000; INTERACTIVE sem suporte na amostra; EXIT 0,066. A matriz balanceada está em `85_eval/confusion.csv`, com prévia em `85_eval/percepcao.png` e contagens em `85_eval/perception_manifest.json`.

### Inspeção visual dos GIFs

- Nível 3: o agente avança até a parede/obstáculo elevado e parece ficar preso ali, sem demonstrar salto consistente para superar a estrutura.
- Nível 4: o personagem permanece no fundo do buraco e não completa a subida pela borda.
- Nível 6: aproxima-se da parede quebrável, mas não há evidência visual de que ataque e atravesse o bloqueio.
- Nos três GIFs, o progresso é pequeno ou ausente; não há evidência de desvio consistente de perigos.
- Os GIFs estão em `85_eval/gifs/nivel_3.gif`, `nivel_4.gif` e `nivel_6.gif`.

### Limites observados

As metas de sucesso do Prompt 85 não foram atingidas uniformemente: níveis 0–2 ficam abaixo de 90%; níveis 3–4 ficam abaixo de 75% mesmo no modo estocástico; nível 7 fica abaixo de 30%. A percepção também fica abaixo das metas para SOLID, HAZARD, ENEMY e PLAYER. O nível 6 oscila muito entre modos, com 18% determinístico e 72% estocástico. O pré-treino terminou pelo limite de 8 horas, em 697.088 passos, bem antes dos 20 milhões configurados, e usou CPU apesar da RTX 3060 disponível; não houve medição de pico de VRAM nem comparação CUDA/AMP.

## Validação

- Suíte completa após a correção do gerador: **784 passaram, 3 ignorados, 3 avisos e 8 subtestes passaram** (`243,45 s`). Os avisos remanescentes são conversões de tensores do Stable-Baselines3.
- `ruff check src tools tests`: passou.
- `mypy src`: passou (98 arquivos).
- `git diff --check`: passou.
- Smoke de treino recorrente com ambientes subprocessados e currículo: concluído previamente até 25.088 passos, incluindo resets entre níveis.
- Amostragem de experiência a 1% e carregamento do checkpoint universal foram verificados previamente.

## Limitações e próximos passos

A campanha ficou aquém de 20M passos e avançou apenas até o nível de currículo 1; portanto, não há evidência suficiente para afirmar que a política passa com folga nos níveis 0–6. O modelo foi treinado em CPU apesar da RTX 3060 prevista pelo perfil de hardware, e a avaliação CUDA/AMP ainda não foi feita. Os prompts 86/87 devem tratar este checkpoint como baseline experimental: os critérios de desempenho do 85 não foram alcançados.

## Arquitetura e hiperparâmetros

O encoder recebe `uint8 (B,12,108,192)` (quatro quadros RGB), aplica blocos IMPALA de 16/32/32 canais e produz mapa `(B,32,14,24)` e vetor ReLU de 256. A cabeça semântica emite logits `(B,10,27,48)`. Vetor visual (256), embedding do histórico de ações (64) e extras do jogo (256; zerados no ginásio) entram na LSTM de 256 unidades. A cabeça `explore` gera logits mascarados sobre as ações universais; o crítico recebe a LSTM e, separadamente, um vetor privilegiado de 64 dimensões. As cabeças auxiliares predizem dinâmica inversa e movimento do herói.

Configuração final registrada em `runtime/v2/pretrain/pretrain-resume-2026-10-23/config.json`: PPO recorrente, rollout 256, batch 512, 4 épocas, LR `2,5e-4`, gamma `0,995`, GAE `0,95`, entropia `0,01`, clip `0,2`, sequência 32; coeficientes semântico/inverso/movimento `0,5/0,1/0,1`, com aumento de deslocamento aleatório. Foram usados 3 ambientes e 3 threads de CPU. O ciclo foi limitado a 8 horas.
