# Prompt 85 — política universal e pré-treino

## Arquitetura entregue

- Encoder IMPALA: entrada empilhada (B,12,108,192), canais 12→16→32→32, dois blocos residuais por estágio; mapa (B,32,14,24) e vetor projetado de 256 dimensões.
- Cabeça semântica: (B,32,14,24) → logits (B,10,27,48), com 255 ignorado. Pesos calibrados em 50.000 quadros de treino; valores em src/voidgym/models/semantic_weights.json.
- Política: histórico 64, concatenação 320, LSTM 256, ator explore de 43 ações com máscara, crítico com informação privilegiada somente na via de valor. Cabeças auxiliares predizem dinâmica inversa e (vx, vy, on_ground).
- PPO recorrente usa GAE, clipe 0,2, sequências de 32 passos, deslocamento pareado imagem/semântica e brilho/contraste. Pesos iniciais: semântica 0,5; inversa 0,1; movimento 0,1.
- Checkpoints atômicos em diretórios por componente, com validação da tabela nominal de ações.

## Validação

- Ruff: passou (ruff check src tools tests).
- Mypy: passou (mypy src/voidgym tools/v2, 52 arquivos). mypy src mantém os dois diagnósticos preexistentes de pyvips em src/hk_agent/map_art.py:27.
- Suíte v2: 52 passaram, 2 pulados; temporário curto no Windows para evitar limite de caminho. E2E: 2 passaram.
- Baseline v1 da fila (286 passed, 10 failed, 409 errors, 0 skipped, 8 subtests): mantido; src/hk_agent/ não foi alterada.
- Micro fumaça da implementação recorrente: 4 passos, 2 ambientes, atualização concluída, perdas finitas; retomadas validadas de 4 para 8 e de 8 para 12 passos. A execução final de uma hora retomou do passo 12 e ultrapassou os 20.000 passos, salvando checkpoint. A tentativa preliminar interrompida em 7.808 foi substituída.

## Pré-treino e avaliação

Máquina Windows 10, Intel Core Ultra 7 258V, 8 núcleos físicos/lógicos, 31,5 GiB RAM, sem GPU NVIDIA; VRAM de pico não aplicável. Plano automático sem reserva para jogos: 6 ambientes e 7 threads; execução longa limitada manualmente a 2 ambientes, 7 threads, CPU e prioridade BELOW_NORMAL.

Calibração semântica: 50.000 quadros, frequências e pesos por classe gravados no JSON de pesos.

Execução CPU recorrente iniciada em 21/10/2026 às 14:04:38 e encerrada naturalmente às 15:04:46 (hora local), por limite de uma hora. Retomou o checkpoint de 12 passos e concluiu em 22.284 passos (22.272 novos); duração 1:00:08, média de 6,18 passos/s no último update. Usou 2 ambientes, lote 128, uma época, rollout 64, sequência 32, gamma 0.995, lambda 0.95, lr 2.5e-4, entropia 0.01, clipe 0.2, c_sem 0.5, c_inv=c_mot=0.1; coletor 1 thread, PPO 7 threads, prioridade `BELOW_NORMAL`. Métricas finais: L_ppo 374,69; L_semantic 1,073; L_inverse_dynamics 2,861; L_hero_motion 0,411; entropia 2,746; KL 0,0016; explained variance -0,016. Currículo terminou no nível 2; sucesso recente 100% no nível 0, 100% no 1 e 70,4% no 2. O treino cumpriu os 20k da fumaça, mas o treino longo de 20M/8h fica pendente para a RTX 3060. Curvas: `relatorios/85_eval/curvas.png`.

Benchmark limpo de 2.000 passos, sem treino concorrente: 3.501 passos/s num processo e 3.615 passos/s agregados com 6 workers. A amostra curta sob carga (912/749 passos/s) fica apenas como referência histórica. Não havia processo v1 para medir.

Avaliação holdout (`split=test`) concluída: 100 episódios por política nos níveis 0–6 e 30 conjuntos por política no nível 7 (1.460 episódios). Tabela completa: `relatorios/85_eval/tabela.md`. Semântica: 5.000 quadros iniciais de salas procedurais holdout, 625 por nível (0–7), com pilha do mesmo quadro em quatro instantes; matriz em `perception_metrics.json` (6.480.000 pixels válidos). IoU por classe: EMPTY 0,893; SOLID 0,419; PLATFORM, HAZARD, ENEMY, PLAYER, BREAKABLE, PICKUP e INTERACTIVE 0; EXIT 0,023. O modelo ainda colapsa a maior parte dos rótulos em EMPTY/SOLID; as metas semânticas não foram atingidas.

| Nível | Estocástica | Determinística |
|---:|---:|---:|
| 0 | 100% | 100% |
| 1 | 100% | 100% |
| 2 | 50% | 88% |
| 3 | 99% | 11% |
| 4 | 27% | 68% |
| 5 | 100% | 100% |
| 6 | 100% | 100% |
| 7 (30 conjuntos) | 96,7% | 76,7% |

Inspeção visual dos GIFs (primeiro episódio determinístico capturado em cada nível), em cinco observações:

1. Nível 3: alcança a parede/obstáculo vertical, mas não consegue subir; a trajetória fica praticamente parada até o limite.
2. Nível 4: faz uma tentativa de salto e alcança a plataforma elevada; o exemplo capturado termina rapidamente.
3. Nível 6: salta em direção às plataformas em camadas e progride pela sala; a amostra visual é curta.
4. Nos níveis simples e no 5–6 a taxa medida é alta, mas os resultados de 3–4 variam bastante entre política estocástica e determinística.
5. A combinação dos GIFs e das métricas mostra que ainda não há comportamento de plataforma robusto nem percepção semântica confiável.

## Checkpoint e próximos passos

Checkpoint final, com métricas de avaliação no manifesto (fora do Git): `C:\Users\thayr\AppData\Local\VoidGym\runtime-v2\pretrain\pretrain85-recurrent-cpu\universal-000022284`. ZIP para cópia manual ao PC principal: `C:\Users\thayr\AppData\Local\VoidGym\runtime-v2\pretrain\pretrain85-recurrent-cpu\universal-000022284.zip`, 13.647.901 bytes. A primeira execução do avaliador completou as métricas, mas o sandbox barrou a gravação do manifesto fora do repositório; o manifesto foi atualizado com a autorização do ambiente e o ZIP foi refeito.

No PC principal: repetir pré-treino com CUDA/RTX 3060, conferir VRAM e throughput, validar jogo real/tempos do ginásio, rodar avaliação no PC principal e seguir para o prompt 86. Nenhuma alegação de transferência para Hollow Knight é feita nesta etapa.
