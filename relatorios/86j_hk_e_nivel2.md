# 86j — bridge do Hollow Knight e falha do nível 2

## A. Bridge do Hollow Knight

### Causa e correção

O probe do 86i encerrava em `env.reset` porque o subprocesso criado com `python -m hk_agent.episode_worker` não recebia o `src` do projeto em seu `PYTHONPATH`. O stderr registra `ModuleNotFoundError: No module named 'hk_agent'`; o supervisor depois reportava `worker result unavailable: FileNotFoundError`. O processo do jogo/mod chegou a iniciar, mas não houve HELLO nem tentativa de conexão ao socket. Não foi uma recusa de capacidade.

`tools/v2/probe_hk_policy.py` agora inclui os caminhos absolutos do projeto e de `src` no `PYTHONPATH` herdado pelo worker. O teste novo executa o módulo a partir de um diretório de trabalho alheio ao repositório, para reproduzir o contexto do subprocesso.

O probe 86j carregou `universal-012438528` na GPU. A DLL de `hk-ai-1` tinha o marcador/API `1.5.78.11833-77` e SHA-256 `ec7799b522d3494b8ef6950d5589c448ef8ae3f524a8e915a3549cb928d07fc6`. O bridge pediu `semantic_labels_v1`, `combat_events_v1` e rótulos de objetos quebráveis. O servidor usou `127.0.0.1` e porta dinâmica do sistema (`port=0`); o número efetivo e o token não foram persistidos. O `ModLog` registra `Supervisor handshake complete`, provando que o mod conectou à porta passada ao jogo e concluiu o handshake. Não houve rejeição das capacidades.

### Probe de cinco minutos

- duração: 305,187 s; 1.448 decisões com posição registrada (cobertura 100%);
- cena: `Tutorial_01`; x inicial 35,460, x final 83,415; avanço líquido para a direita 47,955 unidades;
- movimento inferido da velocidade: 259 decisões para a direita, 106 para a esquerda, 1.083 sem movimento. O deslocamento positivo acumulado foi 398,081 unidades, mostrando bastante recuo apesar do avanço líquido;
- 490 ações de ataque, probabilidade média de ataque 0,349 e 10 eventos `hit_enemy`;
- save da instância restaurado e hash antes/depois igual. O save pessoal e a baseline não foram usados.

O GIF de 30 s está em [`86j_hk_policy_probe/policy-30s.gif`](86j_hk_policy_probe/policy-30s.gif); decisões completas em [`86j_hk_policy_probe/decisions.jsonl`](86j_hk_policy_probe/decisions.jsonl) e resumo em [`86j_hk_policy_probe/summary.json`](86j_hk_policy_probe/summary.json).

## B. Isolamento do nível 2

Foram avaliadas 200 salas por variante e modo, com o mesmo conjunto de seeds do 86i (`28600 + 10 × índice`, 28600–30590), checkpoint `universal-012438528`, limite de 1.500 passos por episódio e dispositivo CUDA. As variantes eram: original; remoção da parede mantendo o vão; preenchimento do vão mantendo a parede; parede de chão com largura de 2 u mantendo o vão; original com ataques mascarados.

“Ataque/ação” é a proporção de ações ATTACK/NAIL_ART sobre todos os passos da avaliação. Distância é a média ponderada por ataque até a parede original; na variante sem ataque não se aplica. “Parado” é o tempo médio por episódio com velocidade horizontal abaixo de 0,08 junto à parede.

| Variante | Modo | Sucesso | Ataques/ação | Episódios com ataque | Distância média ao atacar (u) | Parado junto à parede (s/episódio) | Passos médios |
|---|---|---:|---:|---:|---:|---:|---:|
| Original | Determinístico | 101/200 (50,5%) | 81,7% | 195/200 | 118,26 | 7,86 | 638,2 |
| Original | Estocástico | 112/200 (56,0%) | 62,5% | 199/200 | 78,16 | 0,48 | 673,6 |
| Só o vão (parede removida) | Determinístico | 90/200 (45,0%) | 83,7% | 199/200 | 127,49 | 0,05 | 753,5 |
| Só o vão (parede removida) | Estocástico | 82/200 (41,0%) | 62,5% | 200/200 | 80,03 | 0,21 | 895,8 |
| Só a barreira (vão fechado com chão) | Determinístico | 117/200 (58,5%) | 0,6% | 115/200 | 5,07 | 56,90 | 133,5 |
| Só a barreira (vão fechado com chão) | Estocástico | 197/200 (98,5%) | 50,4% | 194/200 | 59,23 | 0,52 | 36,9 |
| Barreira de chão larga (2 u), vão mantido | Determinístico | 87/200 (43,5%) | 82,6% | 195/200 | 119,83 | 5,07 | 758,4 |
| Barreira de chão larga (2 u), vão mantido | Estocástico | 106/200 (53,0%) | 62,5% | 200/200 | 79,04 | 0,41 | 718,0 |
| Sem ataque | Determinístico | 15/200 (7,5%) | 0% | 0/200 | — | 14,92 | 1.238,7 |
| Sem ataque | Estocástico | 89/200 (44,5%) | 0% | 0/200 | — | 1,05 | 841,0 |

O resultado favorece o vão como obstáculo dominante: remover a parede sem mexer no vão não melhora o sucesso; ampliar/retexturizar a parede tampouco; fechar o vão eleva o sucesso estocástico para 98,5%. Mascarar ataque também não resolve. Isso não confirma a hipótese de confusão da parede fina com quebrável.

Os ambientes/seeds são pareados entre as variantes. A ação estocástica, porém, não usa a mesma sequência aleatória por sala: a avaliação focada foi iniciada com uma nova sequência Torch, enquanto o teste 86i avaliava os níveis anteriores antes do nível 2. Por isso a comparação estocástica do original (56,0% aqui, 60,5% no relatório 86i) é descritiva, não uma reprodução bit a bit. O resultado determinístico original reproduziu exatamente 50,5% do 86i.

Resultados por episódio e resumo de máquina estão em [`86j_level2_variants_28600/variants.jsonl`](86j_level2_variants_28600/variants.jsonl) e [`86j_level2_variants_28600/summary.json`](86j_level2_variants_28600/summary.json). O avaliador reexecutável é [`../tools/v2/eval_86j_level2_variants.py`](../tools/v2/eval_86j_level2_variants.py).

## C. Ajuste de gap, currículo e smoke

A faixa gerada do vão de nível 2 foi reduzida de 35–70% para 35–50% do alcance teórico do pulo. O alcance continua calculado com run speed, tempo máximo de hold e velocidade inicial vertical, enquanto a ação universal `JUMP_MAX` dura 450 ms por chamada e precisa ser mantida por decisões sucessivas quando o perfil físico requer mais tempo. A margem adicional é deliberada: o experimento apontou o vão como causa e o checkpoint falha em salas que mantêm o vão mesmo sem parede.

Para prática de recuperação, os níveis liberados com sucesso abaixo de 70% na janela móvel de 100 episódios recebem peso 3× em relação a cada nível inferior que está na meta, dentro dos 30% de amostras de níveis abaixo do currículo atual.

O smoke autorizado foi executado em CUDA, separado da campanha principal, com um ambiente e limite rígido de 30 min. Um primeiro smoke de 2.048 passos parou pelo guarda herdado de 3M de passos sem melhora no nível 2; ele não alterou o checkpoint-base. A execução diagnóstica seguinte elevou esse limiar apenas para 3,2M, manteve o limite de 30 min e completou 20.480 passos novos (12.438.528 → 12.459.008) em 5min38s. RAM disponível ao final: 2,30 GiB; commit livre: 6,61 GiB; pico CUDA alocado: 1,21 GiB. O sucesso móvel do nível 2 passou de 42% para 43% (100 episódios). Essa diferença de 1 ponto percentual não é evidência suficiente de melhora; o smoke valida a execução da alteração, mas a política continua abaixo da meta. A execução ficou em `E:\VoidGymRuntime\v2\pretrain\86j-gap-smoke20k-2026-11-03`.

## D. Validação

- Suíte completa: **876 passed, 4 skipped, 8 subtests passed**, 3 warnings preexistentes de construção de tensor no Stable-Baselines3. Basetemp do pytest fora do repositório.
- `tests/test_control_plane.py`: 93 passed em três execuções consecutivas.
- `tests/v2/platform_gym/test_generator.py`: 7 passed com o limite de gap atualizado.
- Ruff: `ruff check .` limpo.
- mypy: `mypy src` limpo, 110 arquivos.
- `git diff --check`: limpo.
- `.test-tmp/` antigo foi deixado intacto por conter artefatos de execuções anteriores; `pyproject.toml` exclui essa pasta dos analisadores/testes.

## Estado para 87

O bloqueio técnico do probe de Hollow Knight está resolvido: o bridge completou o handshake com a DLL e o probe de cinco minutos avançou na Passagem do Rei. O próximo treino no HK pode seguir do ponto de vista de conexão. O ajuste de gap/currículo não foi validado como recuperação de sucesso pelo smoke de 20k; o nível 2 ainda está em 43% e requer mais treino/avaliação depois. Portanto 87 está liberado para a campanha no HK, mas não se deve tratar o problema de desempenho do nível 2 como resolvido.
