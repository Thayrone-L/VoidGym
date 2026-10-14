# Riscos e decisões

## 1. Registro de riscos

Escala: probabilidade e impacto de 1 (baixo) a 5 (alto). A pontuação é apenas priorização inicial e será revisada após o diagnóstico.

| ID | Risco | P | I | Mitigação / portão |
|---|---|---:|---:|---|
| R01 | versão escolhida incompatível com Modding API | 3 | 5 | perfis separados; propor 1.5.78.11833 para desenvolvimento; fixar hashes |
| R02 | input interno não funciona sem foco | 3 | 5 | backends isolados; teste com duas janelas; fallback externo avaliado separadamente |
| R03 | captura bloqueia a thread do jogo | 3 | 4 | primeiro medir; AsyncGPUReadback; taxa limitada; drop explícito |
| R04 | frame e ação ficam dessincronizados | 4 | 5 | sequência, trace, ação aplicada, um comando em voo e transição descartável |
| R05 | save pessoal é alterado | 2 | 5 | backup/hash/allowlist/cloud control; falha fechada |
| R06 | saves de instâncias paralelas colidem | 4 | 5 | paralelismo bloqueado até virtualização comprovada |
| R07 | telemetria privilegiada vaza para o modelo | 3 | 5 | canais separados, allowlist e runs privilegiadas inelegíveis |
| R08 | agente explora falha da recompensa | 5 | 4 | componentes auditáveis, testes adversariais e vídeos amostrados |
| R09 | novidade recompensa animação/partículas | 5 | 4 | estabilidade temporal, memória episódica e testes de farming |
| R10 | 16 GB de RAM limitam múltiplas instâncias | 4 | 4 | medir 1/2/4; buffers compactos; limites e inferência compartilhada |
| R11 | RTX 3060 disputa VRAM com jogos | 4 | 3 | medir por processo; rede pequena; batch adaptado; folga obrigatória |
| R12 | reset rápido produz estado inconsistente | 3 | 4 | processo reset como referência; 20 ciclos; só otimizar depois |
| R13 | logs/vídeo enchem o disco | 4 | 3 | orçamento, rotação, amostragem e parada antes de espaço crítico |
| R14 | PPO não aprende com pixels/baixa coleta | 3 | 3 | baseline, tarefa curta, ablações e opção DQN/recurrent/model-based posterior |
| R15 | conhecimento prévio contamina o agente | 2 | 5 | pesos aleatórios, sem encoder/LLM/VLM/demonstrações e proveniência de dados |
| R16 | resultados não são reproduzíveis | 3 | 4 | manifests, seeds, hashes, checkpoints completos e avaliação separada |
| R17 | mudança visual faz memória confundir lugares | 4 | 3 | confiança, múltiplas referências e grafo probabilístico |
| R18 | antivírus/firewall bloqueia IPC ou binário | 2 | 3 | loopback, build assinado por hash local, logs claros; não desabilitar proteção às cegas |
| R19 | saves/mods se misturam entre versões | 4 | 5 | instalações, raízes, manifests e allowlists separados por `game_profile` |
| R20 | política treinada numa versão falha em outra | 4 | 3 | checkpoint vinculado ao perfil; transferência apenas como experimento explícito |
| R21 | builds autônomas distintas gravam no mesmo AppData | 5 | 5 | rastrear I/O; virtualizar caminho ou usar isolamento de usuário; testar antes de paralelismo |

## 2. Decisões firmes

### D001 — Sem modelo de linguagem no loop

Motivo: latência, custo e risco de conhecimento prévio; controle por frame é melhor tratado por política compacta.

### D002 — Imagem + instrumentação restrita

Motivo: reinício, eventos e segurança se tornam auditáveis sem fornecer mapa/posição à política. Requer confirmação do usuário no início da implementação.

### D003 — Um ambiente antes de paralelismo

Motivo: multiplicar uma integração instável só multiplica falhas e risco aos saves.

### D004 — TCP loopback para protocolo v1

Motivo: interoperabilidade e diagnóstico. Memória compartilhada é uma otimização condicionada a benchmark.

### D005 — Reset de processo como referência

Motivo: mais lento, porém mais fácil de auditar. Reset interno precisa provar equivalência suficiente.

### D006 — Pesos aleatórios

Motivo: preservar o desconhecimento inicial. Qualquer encoder futuro exige análise explícita do risco de contaminação.

### D007 — Versão por perfil, não versão única

Motivo: desenvolvimento favorece compatibilidade de modding; velocidade precisa obedecer à categoria. Proposta: `1.5.78.11833` no perfil de desenvolvimento e `1.2.2.1` apenas no perfil futuro de Any%/NMG correspondente. Instalações e dados permanecem separados.

### D008 — Builds autônomas como fonte do jogo

Motivo: o usuário já possui builds utilizáveis fora da Steam. Steam deixa de ser dependência; o inventário por caminho, versão e SHA-256 vira a fonte de verdade. Isso não altera a exigência de virtualizar ou isolar a pasta de saves compartilhada pelo Unity.

## 3. Decisões condicionais

| ID | Decisão | Evidência necessária |
|---|---|---|
| C001 | versão exata da Modding API/Lumafly no perfil de desenvolvimento | hash do jogo e smoke test |
| C002 | backend de input | controle sem foco, lease e duas instâncias isoladas |
| C003 | captura síncrona ou assíncrona | correção visual, suporte e benchmark |
| C004 | RGB ou luminância | ablação de legibilidade e aprendizado |
| C005 | resolução/taxa/K frames | latência, detalhe e coleta sustentável |
| C006 | PPO versus alternativa | baselines e primeira tarefa |
| C007 | reset interno | equivalência operacional em ciclos repetidos |
| C008 | 1, 2 ou 4 instâncias | aprendizagem/hora, recursos e falhas |
| C009 | mecanismo de novidade | testes adversariais e estabilidade entre tentativas |
| C010 | virtualização de saves | inspeção das APIs/assemblies da versão instalada |

## 4. Critérios de parada segura

Interromper automaticamente quando ocorrer:

- caminho de save fora da allowlist;
- backup ou hash de proteção ausente;
- versão/protocolo incompatível;
- entradas não liberadas no prazo;
- fila de comandos/frames crescendo;
- repetição de frame acima do limite sem marcação;
- espaço em disco abaixo da reserva;
- RAM/VRAM acima do limite configurado;
- crash loop;
- taxa de transições inválidas acima do limite;
- evento desconhecido no canal de aprendizagem.

O sistema não tenta “seguir em frente” quando a integridade da experiência ou dos saves deixa de ser demonstrável.
