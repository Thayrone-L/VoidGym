# ADR 0004: infraestrutura autônoma da v2

## Contexto
A v2 precisa guardar experiência e executar campanhas sem dependência do jogo, respeitando os recursos disponíveis e mantendo os dados reproduzíveis.

## Decisão
Centralizar campanha em configuração versionada com hash, detectar hardware para montar um plano explícito, persistir experiência em blocos NPZ atômicos com índice JSONL e executar gravação em segundo plano. Recompensas universais e específicas ficam separadas; ganhos positivos têm limites por fatia e janela móvel, enquanto o custo temporal permanece contínuo. Validar o ciclo integrado por atores, learner e supervisor em CI Windows.

## Consequências
Campanhas registram configuração, hardware e plano de recursos; a experiência pode ser filtrada e retida por prioridade. A CI não comprova disponibilidade de CUDA nem permissões de Actions do repositório. O teste GPU e o perfil da máquina principal permanecem pendentes.
