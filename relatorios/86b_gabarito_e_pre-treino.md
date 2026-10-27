# Prompt 86b — gabarito semântico e pré-treino CUDA

## Resultado até 2026-10-27 11:24 -03

O trabalho de gabarito, currículo, proteções do pré-treino, smoke tests, validações e deployment está concluído. A campanha principal está **em andamento**; os resultados finais de treinamento e a avaliação pós-treino serão acrescentados quando ela terminar.

## Gabarito semântico

- Captura isolada em `Tutorial_01`, 512 frames, sem movimento, na instância `hk-ai-1`. O save pessoal não foi usado.
- Na captura final corrigida: tempo de captura semântica mediano **0,3627 ms**, p95 **0,42281 ms**.
- Frações dos pixels: EMPTY 91,821%; SOLID 6,019%; PLAYER 0,463%; INTERACTIVE 1,698%; demais classes 0% nesta vista estacionária.
- Gabarito visual: [`86_amostras/hk_gabarito.png`](86_amostras/hk_gabarito.png). A vista inclui chão, paredes, jogador e elementos interativos. Inimigos, espinhos e saída não aparecem nesta composição; portanto, esta captura não valida visualmente essas classes.
- Dry zero-shot do checkpoint inicial `universal-000697088`, sem atualização de pesos: IoU EMPTY **0,8552**, SOLID **0,0111**, PLAYER **0,1482**, INTERACTIVE **0,0**. A métrica de interativos é fraca e requer exemplos variados para uma avaliação conclusiva.

## Pré-treino e perfil

- O checkpoint inicial era CPU-only (`cuda_available:false` no plano antigo), embora o PC tenha NVIDIA. Foi acrescentada uma proteção que recusa treino CPU quando há GPU NVIDIA, salvo `--allow-cpu` explícito.
- O smoke pós-otimização retomou `universal-000697088` e avançou até **717.568**: **20.480 passos efetivos em 102,86 s (199,11 passos/s)**, 10 ambientes, learner e atores em CUDA, pico alocado **4.481 MiB**, reservado **8.542 MiB**.
- Perfil dos 20.480 passos: atualização PPO 63,8% do tempo, coleta 21,8%, inferência 14,3%, GAE 0,1%. O update continua sendo o maior gargalo.
- Smoke de retomada partiu de `universal-000717568` e avançou até **720.128**: mais **2.560 passos em 14,73 s (173,74 passos/s)**, em CUDA.
- Foi feita uma tentativa inicial de smoke com `--total-steps 20480`, menor que o contador do checkpoint. Como o alvo é absoluto, ela encerrou sem novos passos; a tentativa correta subsequente confirmou o avanço acima.
- Campanha principal `pretrain-86b-gpu-2026-10-27`, runtime em `E:\VoidGymRuntime\v2\pretrain\pretrain-86b-gpu-2026-10-27`, retomada **diretamente** do checkpoint original `universal-000697088`. Alvo de 100 milhões de passos, limite de 24 horas, parar ao alcançar nível de currículo 6. O runtime plan confirma 10 ambientes e dispositivos `cuda` para treino e ator.
- Último log consultado às 11:24:05 -03: **725.248 passos totais**, ou **28.160 além do checkpoint inicial**; currículo 2, sucesso recente 1,0. As duas últimas linhas registraram 215,75 e 217,61 passos/s. Processo ativo, sem conteúdo no `stderr.log`. O último checkpoint íntegro disponível para retomada é o original `C:\VoidGym\runtime\v2\pretrain\pretrain-resume-2026-10-23\universal-000697088`; ainda não houve checkpoint periódico novo da campanha principal (salvamento a cada 250 mil passos).

## Validação das políticas simples

Avaliação de 200 mapas de teste por nível; resultados completos em [`86b_politicas_bobas.json`](86b_politicas_bobas.json):

| Nível | Solver | Falha RIGHT | Falha RIGHT+JUMP_TAP | Falha aleatória | Critério |
|---:|---:|---:|---:|---:|---|
| 2 | 100% | 100% | 99% | 99% | Passou |
| 3 | 100% | 83,5% | 90,5% | 91% | Passou |
| 4 | 100% | 100% | 100% | 80% | Passou |
| 5 | 100% | 85% | 95,5% | 99% | Passou |
| 6 | 100% | 100% | 100% | 99,5% | Passou |
| 7 | 100% | 100% | 100% | 100% | Passou |

No nível 7, a porta sem a habilidade falhou em **200/200** tentativas. Os níveis 2, 4 e 5 receberam obstáculos controlados para exigir saltos deliberados; o teste de determinismo, certificação e velocidade do gerador passou.

## Qualidade e deployment

- Suíte completa: **836 passed, 4 skipped, 8 subtests passed**; 3 avisos existentes do Stable-Baselines3.
- Ruff: passou nos arquivos alterados. Mypy: sem erros nos nove arquivos verificados.
- Testes semânticos C# e build Release do mod passaram durante a implementação.
- Deployment oficial já validado na instância isolada `hk-ai-1`: SHA-256 `76e446d47d5e896cf4106a66149e6167296ac33078bc3559cb3e3a422cd25480`; backup em `C:\VoidGym\artifacts\bridge-deploy-backups\bridge-20261027-095355-37e352d7\hk-ai-1`.
- O relatório final receberá as métricas de encerramento, último checkpoint íntegro e avaliação/zero-shot pós-treino depois que a campanha parar por nível 6 ou pelo limite de 24 horas.
