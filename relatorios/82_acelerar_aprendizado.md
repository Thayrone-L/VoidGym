# Prompt 82 — acelerar o aprendizado

Data da revisão: 2026-10-21

## Estado

Implementação parcial, sem experimento de jogo. Nenhum treino foi iniciado ou encerrado nesta revisão. A linha de base de uma hora e as comparações reais ficam pendentes de uma solicitação explícita para treinar.

Foi implementado `augment_random_shift` no caminho de atualização da política recorrente: deslocamento uniforme de até 4 px, borda replicada e deslocamento comum a todos os quadros da mesma sequência LSTM. A augmentação é aplicada por `evaluate_actions`, usado nas atualizações PPO e SIL. A coleta e a predição usam `forward` sem augmentação. O padrão permanece desligado até validação real.

As opções `game_speed` diferente de 1.0, `aux_losses`, `asymmetric_critic` e `mask_interact_without_prompt` ainda não têm integração completa entre campanha, Python e bridge/mod. A configuração agora as rejeita explicitamente, antes de tratar uma opção incompleta como ativa. Nenhuma dessas opções deve ser considerada implementada.

## Auditoria da avaliação

No learner compartilhado, campanhas `explore_v3` e `explore_v4` avaliam a cada 20 rodadas por padrão. Um ator alternado executa uma prova determinística e outra estocástica, cada uma limitada a 1.000 passos (ou ao horizonte menor da campanha). Isso confirma a cadência configurada do prompt 75 para essas versões; não mede o tempo real gasto em espera, avaliações, relançamentos ou stragglers. Outras versões de recompensa não entram nesse agendamento de avaliação do learner compartilhado.

## Medições

| Métrica | Linha de base | Com opção ativa | Estado |
| --- | ---: | ---: | --- |
| Progresso máximo na Passagem do Rei / hora de jogo | Pendente | Pendente | Requer treino autorizado |
| Progresso máximo na Passagem do Rei / hora real | Pendente | Pendente | Requer treino autorizado |
| Travadas por hora | Pendente | Pendente | Requer treino autorizado |
| Passos por segundo | Pendente | Pendente | Requer treino autorizado |
| Entropia | Pendente | Pendente | Requer treino autorizado |
| Retorno das avaliações | Pendente | Pendente | Requer treino autorizado |
| Tempo de espera/avaliação/relançamento/straggler | Pendente | Pendente | Sem observação de campanha ao vivo |

Não encontrei `learner.json`, a pasta `actors/` nem artefatos com o identificador `jogada-1-repeat-1c3b` nos diretórios de estado do projeto. Também não havia processo Hollow Knight visível na consulta local. Portanto, o estado de uma campanha fora desses diretórios não pôde ser confirmado; nenhuma ação foi tomada sobre processos ou campanhas.

## Verificações executadas

- `ruff check` nos módulos e testes alterados: aprovado.
- Testes recorrentes, defaults/validação de campanha e rejeição das opções incompletas: 18 aprovados.
- Avisos: três avisos já emitidos pelo SB3 ao construir tensores a partir de tensores.
- Não executados: linha de base ou comparação real, teste de velocidade e pulo, medição de CPU/quadros, teste da DLL, implantação da DLL ou início/retomada de treino.

## Próximos itens técnicos

1. Integrar `game_speed` no bridge isolado, nas durações e nos relógios de jogo; testar estabilidade antes de permitir valores acima de 1.0.
2. Instrumentar os rótulos de movimento e grounded no bridge e treinar/serializar as cabeças auxiliares sem alimentar a política.
3. Separar o caminho privilegiado do crítico e cobrir a invariância do ator quando o vetor privilegiado muda.
4. Detectar o aviso de interação visível no mod, transmiti-lo no protocolo e só então habilitar a máscara opcional.
5. Depois das integrações e da autorização de treino no momento, executar as medições individuais e escolher os padrões com base na métrica principal.
