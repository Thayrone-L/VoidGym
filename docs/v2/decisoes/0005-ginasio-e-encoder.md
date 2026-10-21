# Decisão: ginásio procedural e treino universal supervisionado

**Data:** 2026-10-21

O ginásio de plataformas fornece ambientes procedurais, física determinística e gabaritos
semânticos independentes do estilo. A política universal recebe somente pixels, histórico de ações
e máscara. A cabeça semântica, dinâmica inversa, movimento do herói e estado privilegiado do
crítico são supervisionados durante treino e ficam fora do caminho de decisão do ator.

O encoder IMPALA produz um mapa convolucional compartilhado e um vetor de 256 unidades. Uma LSTM
mantém contexto temporal; atores nomeados e mascarados podem crescer sem mudar o contrato.
Checkpoints são divididos por componente e validam a tabela de ações antes de carregar.

**Consequência:** treinamento e gabaritos são próprios do VoidGym; validação no Hollow Knight
continua necessária para calibrar duração temporal e transferência.
