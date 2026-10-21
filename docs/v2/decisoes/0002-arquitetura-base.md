# Decisão: bases da arquitetura da v2 (2026-10-21)

Decisões tomadas com o usuário antes de começar a v2. Elas complementam `decisoes/ia-gamer-universal.md`.

| # | Tema | Decisão |
|---|---|---|
| 1 | **GPU** | O operador tem uma **RTX 3060 12 GB**. A regra "PPO só na CPU" da v1 foi escolha do agente, não do usuário. Na v2, o **learner treina na GPU** quando ela existir (venv CUDA separado). Atores, ginásio e jogos ficam na CPU. Sem GPU, tudo cai para a CPU sozinho. A v1 continua na CPU. |
| 2 | **Treino sincronizado** | A v2 mantém as **rodadas sincronizadas** da v1, que já funcionam. O learner é desenhado para trocar depois por treino assíncrono (IMPALA/V-trace), se valer a pena. |
| 3 | **Armazém de experiência** | Um formato único para toda experiência (ator, avaliação, demonstração, melhores tentativas, ginásio): blocos de até 1.024 passos em `.npz` comprimido + índice JSONL. O operador libera **~100 GB**. A retenção nunca apaga demonstrações nem ajudas, apaga por último os eventos raros e recordes, e do resto guarda uma amostra quando enche. Estimativa: 1M de passos ≈ 10–20 GB. |
| 4 | **Configuração** | Um `CampaignConfig` tipado por campanha, validado num lugar só. Chave desconhecida é erro com sugestão. A configuração final é salva com os resultados e tem hash. |
| 5 | **Autoconfiguração pelo hardware** | `detect_hardware` + `plan_resources` decidem instâncias, ambientes, threads, dispositivo, batch e orçamento de disco em qualquer máquina, incluindo as dos **amigos que vão ajudar a treinar**. **O contrato do modelo não muda com o hardware:** imagem **192×108**, 4 quadros, passo de decisão `STEP_GAME_MS` (medido no HK), ações universais e mapa semântico **27×48**. Assim, modelos de máquinas diferentes continuam compatíveis. Jogos com outra proporção de tela entram com faixas pretas. |
| 6 | **Recompensa dividida** | A parte universal (novidade, progresso, dano, morte, tempo sem teto, sala nova, travada) fica no núcleo. A parte de cada jogo fica no adaptador. As duas são gravadas separadamente. Os tetos são por fatia de episódio + janela de tempo, nunca dependendo só de evento do jogo (lição do 67/70). |
| 7 | **Teste ponta a ponta** | Um e2e sem jogo, em menos de 3 min: sobe supervisor + learner + 2 atores, faz 3 rodadas, mata um ator, para e retoma. Roda em todo prompt da v2. CI no GitHub para os testes rápidos. |
| 8 | **Regras de código** | Módulos ≤ 600 linhas e funções ≤ 80. Nada minificado. Configuração, caminhos e variáveis de ambiente num lugar só. Erros claros em português. `docs/v2/arquitetura.md` é atualizado a cada prompt, com ADRs curtos em `docs/v2/decisoes/`. Tudo verificado por testes automáticos. |

## Para depois

- **Painel da v2:** por enquanto, o painel da v1 com a opção "Motor v1/v2".
- **Captura mais rápida no HK:** memória compartilhada.
- **Instalador para colaboradores:** usa o perfil de hardware.

## Onde isso entra

- O prompt **83b** implementa os itens 3–8.
- O **85** usa a GPU.
- Os prompts **83, 84, 85 e 86** já usam 192×108 e 27×48.
