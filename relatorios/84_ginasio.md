# Prompt 84 — ginásio de plataformas v2

**Estado: implementação e validação técnica concluídas; commit/PR pendentes.** O ginásio, o adaptador e as ferramentas foram implementados. Não iniciei nem parei treinos da v1, não abri Hollow Knight e não alterei `src/hk_agent/`.

## Implementado

- `STEP_GAME_MS = 80 ms`: a v1 usa `lease_ms=80` e `action_repeat=3`; o contrato v2 mantém 80 ms por ação da política, subdivididos em quatro passos físicos de 20 ms.
- Física AABB determinística com pulo variável, coyote time, buffer, pulo duplo, dash, plataformas one-way, ataque/pogo, dano, respawn, pickups, banco, porta e saída.
- Níveis 0–7. Níveis 0–6 têm salas resolvíveis, incluindo vãos recuperáveis (2), saída de buraco (4) e parede quebrável obrigatória (6). Há saídas verticais, espelhamento, prateleiras e tetos. O nível 7 cria campanhas conectadas de 3–6 salas, com habilidade, porta e banco.
- Gerador com prova por solver e novas tentativas. A variante de sala alta é habilitada nos níveis 0–6 fora das campanhas conectadas; foi desabilitada nos conjuntos do nível 7 porque as rotas verticais falharam na prova da cadeia.
- `PlatformGymAdapter` usa as ações universais do HK, mascara ações indisponíveis e integra com `AdapterEnv`. Recompensa combina exploração universal e eventos do ginásio.
- Render RGB 192×108, texturas determinísticas, paletas, fundos em camadas, partículas, iluminação, HUD e câmera; gabarito semântico 27×48 com prioridade e movimento do herói.
- Currículo serializável, preview de mosaico/gabarito e treino/teste, GIFs do solver e benchmark abaixo da prioridade normal.

## Validação

- Suíte completa após a última alteração visual: **775 passaram, 3 ignorados, 8 subtestes passaram**, em 211,82 s. Restaram apenas avisos do Stable-Baselines3 sobre conversão de tensores.
- `tests/v2/platform_gym` e E2E: **25 passaram**, em 70,50 s. `ruff check src tools tests`, `mypy src` (90 arquivos) e `git diff --check` passaram após a mudança visual.
- `ruff check src tools tests`: passou.
- `mypy src`: passou (90 arquivos).
- `git diff --check`: passou.
- Benchmark, prioridade `BELOW_NORMAL`, 3.000 quadros por worker, nenhum processo Hollow Knight observado: **1.944,7 quadros/s em um processo** (meta: 1.500); com 14 workers, **11.609,1/s agregados, 829,2 por worker**. A meta do prompt é por núcleo em uma execução; a taxa cai quando 14 processos competem pela máquina.
- O gerador foi medido em 200 seeds por nível. Média geração + solver: nível 0 **33,8 ms/sala**, 1 **26,6**, 2 **11,4**, 3 **23,0**, 4 **9,4**, 5 **28,8**, 6 **43,2**. O teste verifica 100% de sucesso nos seeds exercitados; campanhas de nível 7 também são verificadas pela suíte.

## Inspeção visual

As amostras foram regeneradas e inspecionadas:

- `relatorios/84_amostras/mosaico.png`
- `relatorios/84_amostras/gabarito.png`
- `relatorios/84_amostras/teste_vs_treino.png`
- `relatorios/84_amostras/solver_nivel_{3,4,6,7}.gif`

As paletas, texturas, fundos, personagens e objetos têm variação visível entre amostras, e a divisão treino/teste apresenta formas reservadas. O fundo agora inclui silhuetas determinísticas de blocos urbanos, cristais, arcos ou formas orgânicas, sem afetar física nem labels. O gabarito marca sólidos, perigos, inimigos, pickups, saídas e o jogador. A geometria jogável continua reconhecível como o mesmo gênero de plataforma e repete corredores horizontais; as cenas parecem mais variadas pela combinação de paleta, textura e cenário, embora ainda não representem gêneros de jogo diferentes. As variantes de salas altas existem, mas são raras (8% nos níveis 0–4 e 6; 2% no 5). A tentativa de aumentar sua frequência e a de plataformas alcançáveis quebrou seeds do solver/tempo; ambas foram retiradas.

## Estado do repositório e pendências

O checkpoint do Prompt 84 foi revisado e registrado no commit `a2e0767` (`v2: procedural platform gym with domain randomization and semantic labels`), já enviado ao PR #8. Nenhum arquivo de `src/hk_agent/` foi alterado. A variação visual de cenário foi ampliada e a meta de throughput e a suíte completa passaram. A frequência baixa de salas altas permanece como limitação documentada.
