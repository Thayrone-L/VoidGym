# Memória recorrente e máscara de ações (etapa 62)

A exploração agora seleciona `MaskableRecurrentPPO`: extrator Dict atual (imagem e conhecimento), LSTM configurável (padrão 256 unidades, uma camada), depois cabeças de política e valor. A mesma distribuição categórica mascarada é aplicada na coleta e durante as atualizações PPO. O adapter SB3 estende `RecurrentPPO`; buffers salvam action masks, estados ator/crítico e episode starts. O learner compartilhado reconstrói estados por sequência e treina BPTT truncado com comprimento configurável (padrão 128).

Estados atravessam os limites de rollout/chunk PPO. O reset ocorre em `episode_start` do ambiente, que representa fim de episódio/morte, reabertura do jogo, recarga do banco ou troca de save. Warm-start safetensors valida hash, fingerprint e manifest e copia exclusivamente tensores compatíveis do extrator; LSTM e cabeças ficam novos. Checkpoints recorrentes podem ser retomados.

O rollout distribuído carrega máscaras, episode starts e estados pi/vf; leitura valida tipo, forma e valores. A publicação da política continua usando o mecanismo imutável de snapshot do learner.

Validação: teste sintético de pista atrasada aprendeu ≥95% no fim da sequência, embora os frames finais fossem idênticos; testes cobrem máscara, persistência/reset, serialização, buffer/learner, retomada, shared rollout e compatibilidade de distribuição. `ruff` e `mypy src` passam; pytest completo: 611 passed, 2 skipped, 8 subtests. SB3 e sb3-contrib fixados em 2.9.0.

Medição CPU com 1 thread, feature vector 256, 18 ações, LSTM 256, 20 aquecimentos e 200 chamadas: núcleo recorrente mean 0.507 ms, p95 0.698 ms por passo. Isso mede o núcleo após extração de features, não a captura de tela/CNN inteira. O custo de uma rodada sintética do learner é registrado pelo próprio trainer (`train_seconds`); o teste de integração completa a coleta e atualização PPO.

Smoke de jogo: tentativa pelo runner transacional `run_isolated_session.py` retornou "Session worker did not produce a result" (transaction exit code 0); não iniciou processo de jogo nem escreveu resultado. Logo a validação em jogo permanece pendente. Nenhum save pessoal/baseline foi modificado; não iniciar campanha até repetir o smoke com sucesso. Estado: AMARELO por smoke pendente.
