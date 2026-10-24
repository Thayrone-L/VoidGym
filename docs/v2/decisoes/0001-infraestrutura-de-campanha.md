# ADR 0001: Infraestrutura única de campanha

- **Contexto:** a v2 precisa reproduzir campanhas em máquinas diferentes, persistir experiência
  em um formato comum e permitir testes sem Hollow Knight.
- **Decisão:** cada campanha usa configuração congelada e com hash; o plano de hardware é
  registrado; transições são gravadas em blocos NPZ atômicos; recompensa universal fica
  separada da recompensa específica do adaptador. O primeiro E2E usa processos e um adaptador
  falso, sem iniciar o jogo.
- **Consequências:** testes e resultados ficam reproduzíveis sem depender da instalação do jogo.
  O harness de learner/ator é mínimo e será substituído/integrado pelos prompts 85–87; nenhum
  resultado sintético representa desempenho de treinamento real.
