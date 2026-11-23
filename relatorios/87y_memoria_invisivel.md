# 87y — investigação da memória não atribuída

## Escopo e segurança

- A campanha `pretrain-87v-beco-2026-11-11-cuda-r6` terminou em 12/11/2026 às 15:13:28, com falha do guard por commit livre abaixo de 1,5 GiB durante 1.800 s. Não houve erro CUDA.
- Em 23/11/2026, a consulta de processos não encontrou `pretrain_platform.py` nem `supervise_pretrain.py` em execução. Nenhum aplicativo foi fechado; a leitura ao vivo mostra Codex, Claude e outros processos do sistema/usuário.
- Esta coleta é somente leitura. O coletor de uma hora grava snapshots no runtime r6 e não interage com os aplicativos.

## Histórico r6

O arquivo `memory-history.jsonl` tem 14 amostras entre 14:02:29 e 15:09:00. O commit livre caiu de 10,92 GiB no início para 2,32 GiB em 14:07; ficou abaixo de 1,5 GiB em 9 das 11 amostras seguintes e encerrou em 1,08 GiB. A estimativa de commit potencial permaneceu entre 16,81 e 18,35 GiB nas amostras finais.

O histórico guarda apenas os dez maiores processos, não a lista completa nem RAM disponível. A soma privada desses dez foi 2,06 GiB na primeira amostra, chegou a 4,63 GiB e terminou em 3,00 GiB. A diferença entre commit usado e essa soma não identifica um processo oculto: inclui todos os demais processos e categorias de memória que não são memória privada dos dez listados. Portanto, não atribuo os cerca de 27,72 GiB de diferença final a driver ou serviço específico.

O registro terminal do guard, mais completo que o histórico periódico, mostra às 15:13:28: RAM disponível 4,57 GiB; commit usado 30,76 GiB; limite 31,80 GiB; commit livre 1,03 GiB; capacidade potencial livre 17,03 GiB. O learner tinha 1,29 GiB de memória privada e os dois workers cerca de 0,025 GiB cada. PyTorch reportava 0,095 GiB alocados e 6,82 GiB reservados na GPU. Essa reserva CUDA não explica o commit do Windows e não é evidência de OOM.

## Snapshot de 23/11/2026, 19:30:22 -03

- RAM total: 15,80 GiB; disponível: 4,07 GiB.
- Commit: 19,05 GiB usados de 31,80 GiB; 12,76 GiB livres. Capacidade potencial livre estimada: 28,81 GiB.
- `Get-Process` enumerou 356 processos acessíveis, com 15,00 GiB de memória privada somada. O commit usado excede essa soma em 4,04 GiB. Esse residual não é uma atribuição causal: inclui pools do kernel, páginas compartilhadas/mapeadas e processos ou componentes não representados por memória privada.
- Maiores processos por memória privada: NVIDIA Broadcast 0,86 GiB; Opera 0,71 e 0,64 GiB; dois WebView 0,57 e 0,56 GiB; MsMpEng 0,48 GiB; Explorer 0,42 GiB; ChatGPT 0,38 GiB; Opera 0,36 GiB; DWM 0,33 GiB.
- GPU: 1.798 MiB usados de 12.288 MiB. `nvidia-smi` listou processos WDDM com uso por PID como `[N/A]`; não dá para atribuir a VRAM a um processo específico.

## Coletor e validação

`tools/v2/diagnose_memory.py` registra, em JSONL, memória física, commit, contadores PerfOS, todos os processos visíveis por memória privada/working set e leitura do `nvidia-smi`. O Windows fornece `CommittedBytes` e `CommitLimit` pelo CIM quando `GetPerformanceInfo` não está acessível; os campos potencialmente desconhecidos permanecem `null`. O coletor não fecha processos nem muda prioridade, pagefile ou configurações.

Foi iniciado às 19:35:27 um coletor oculto, com 13 amostras espaçadas em 300 s (60 min do primeiro ao último), em `E:\VoidGymRuntime\v2\pretrain\pretrain-87v-beco-2026-11-11-cuda-r6\memory-history-87y-2026-11-23.jsonl`. O relatório final da série fica pendente até a última amostra. A leitura inicial independente indicou 12,76 GiB de commit livre, acima do mínimo de 6 GiB exigido para comandos mais pesados.

### Fases controladas — 23/11/2026

As medições foram feitas sem iniciar a campanha, sem salvar modelo/checkpoint e sem fechar Claude ou outros aplicativos. Entre as fases houve 120 s de espera. O commit livre nunca caiu abaixo do piso de 6 GiB.

| Fase | Commit livre | RAM disponível |
|---|---:|---:|
| Base sem PyTorch | 11,74 GiB | 3,88 GiB |
| PyTorch importado, CUDA disponível | 11,11 GiB | 3,64 GiB |
| Política criada na GPU | 10,49 GiB | 3,47 GiB |
| Uma atualização PPO sintética na GPU | 9,16 GiB | 2,48 GiB |
| Política GPU liberada | 8,77 GiB | 2,80 GiB |
| Dois workers de ambiente ativos | 7,99 GiB | 2,73 GiB |
| Workers encerrados | 9,04 GiB | 2,81 GiB |
| Uma atualização PPO sintética em CPU | 8,39 GiB | 2,51 GiB |
| Política CPU liberada | 8,81 GiB | 2,73 GiB |

O smoke test CUDA separado capturou o alocador do próprio processo: política criada com 15,2 MiB alocados/36 MiB reservados; pico durante a atualização sintética de 86,4 MiB alocados/124 MiB reservados. O `nvidia-smi` marcou 2.536 MiB globais após o teste, mas não separou o uso do processo no modo WDDM. Esse batch de dois passos serve para validar o caminho e medir o custo mínimo, não para prever a memória do batch completo de pré-treino.

Validação do coletor: Ruff passou; sintaxe Python passou; as três funções de teste foram executadas diretamente e passaram. `pytest` não está instalado no `.venv-cuda`, então não foi executado via pytest nem foram instaladas dependências. A série horária ainda está em andamento.

## Interpretação

O r6 falhou porque o commit corrente ficou abaixo do limite configurado; a memória potencial de crescimento do pagefile permaneceu alta, mas esse potencial não satisfaz o guard de commit atual. O histórico top-10 não permite concluir qual processo ou componente causou a maior parte do commit. As medições novas incluem a soma de todos os processos visíveis para reduzir essa lacuna, mas ainda não distinguem commit do kernel, páginas compartilhadas e componentes fora da enumeração. Atribuição a driver/serviço específico requer evidência adicional, não apenas a diferença entre commit e memória privada.

