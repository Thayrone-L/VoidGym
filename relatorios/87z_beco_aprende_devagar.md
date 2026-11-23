# 87z — medir e decidir sobre o aprendizado do beco

## Estado e origem dos dados

- O pré-treino r6 terminou em **FAILED** em 12/11/2026 às 15:13:28 -03, após 1.800 s com commit livre abaixo de 1,5 GiB. Nenhum learner continua ativo em 23/11/2026.
- Origem: checkpoint íntegro `universal-012739584`, do r4. Base comparável do projeto: `universal-012438528`.
- No último heartbeat do r6: 12.983.296 passos totais, currículo no nível 8, success_100 dos níveis 0–9: 92%, 99%, 45%, 98%, 97%, 97%, 94%, 97%, 99%, 16%. O nível 9 não atingiu 70%.
- Duração de parede do r6: 4.260,625 s (1h11m00,625s), incluindo espera por memória. Avanço desde a base `universal-012438528`: 544.768 passos. Último checkpoint íntegro: `universal-012940288`, ou 501.760 passos acima da base; os 43.008 passos seguintes não foram salvos.

## Etapa 1 — exposição, progresso e falha

Entre o checkpoint de origem r4 (`universal-012739584`, 9.379 episódios de currículo, histórico do nível 9 com 201 episódios) e o checkpoint r6 (`universal-012940288`, 9.541 episódios, histórico do nível 9 com 300 episódios), houve 162 episódios de currículo, incluindo 99 do nível 9. No intervalo de 30,72 min até o último checkpoint, isso corresponde a 193,4 episódios do nível 9 por hora; o nível 9 representou 61,1% dos episódios adicionados. Portanto, durante esse trecho o beco **não** recebia apenas 15% dos episódios: era o nível corrente e a regra de currículo o tratava como tal.

O arquivo não registra transições completas por nível. Com 99 episódios do nível 9 e limite de 1.500 passos por episódio, o máximo teórico desse trecho seria 148.500 de 200.704 passos (74%); a fração real de passos do nível 9 não é medida e pode ser menor. `experience/index.jsonl` contém três blocos de replay amostrados (2.438 transições no total), não trajetórias completas.

O sistema promoveu o currículo para o nível 9 e depois recuou para o nível 8 quando a janela de 300 episódios do nível 9 ficou abaixo do limite de sucesso de 20% para permanência. A janela final de 100 episódios registrou 16% no nível 9. Isso aponta para baixa taxa de conclusão após exposição substancial, não para falta comprovada de episódios.

Não há logs episódicos suficientes para contar: episódios travados sem recuo, recuos de pelo menos cinco unidades, primeira escada ou plataforma alcançada, nem fração de retornos positivos. O valor de `quick_eval_success` está vazio em todas as 118 linhas de `log.csv`; não houve avaliação rápida registrada nesse r6. Os cinco últimos snapshots de rollout de nível 2 foram 44%, 44%, 44%, 44% e 45%, mas não são cinco avaliações rápidas independentes. O limiar da opção D, portanto, não foi satisfeito pelo critério de cinco quick evals.

O retorno global também não pode ser reduzido a “só recompensa na saída”: `CombinedPlatformReward` combina `UniversalReward`, que inclui sinais de progresso, dano, morte e tempo; `GameReward` ainda dá bônus de saída. Sem retornos por episódio e seus componentes, não é possível medir qual sinal dominou o aprendizado.

## Etapa 2 — alternativas

| Opção | Evidência atual | Custo/risco | Decisão recomendada |
|---|---|---|---|
| **A. Demonstrações do solver** | O solver certificou 200/200 salas do split de teste; as políticas bobas falharam em 199/200. Não há dado de reprodução de ações do solver pela política. | Preparar pares observação/ação; risco de sobreajuste ou interferência nos demais níveis. | Melhor primeira hipótese para teste offline. Só considerar eficaz se a sequência traduzida for reproduzida em pelo menos 90% de um conjunto separado de salas do treino; medir também splits de teste. Nenhuma alteração do treino está autorizada ainda. |
| **B. Recompensa de progresso limitada** | O reward composto já inclui shaping de progresso. Faltam retornos por episódio e decomposição das recompensas. | Pode reforçar recuos, exploração improdutiva ou atalhos. | Não aplicar até registrar trajetória e componente de recompensa por episódio; usar apenas se a falha dominante justificar. |
| **C. 25–30% dos passos para beco** | A parcela de episódios no nível 9 foi 61,1% no trecho mensurável. A parcela de passos não foi registrada. | Pode deslocar capacidade de outros níveis e não resolver o comportamento que impede a conclusão. | Não aplicar com os dados atuais. Primeiro instrumentar passos por nível; reavaliar somente se a participação real ficar abaixo do limite acordado. |
| **D. Limitar nível 2 a 5%** | Últimos cinco snapshots de rollout ficaram em 44–45%, mas não são quick evals. | Pode deixar a política abaixo do nível 2 sem progresso. | Manter pendente; considerar após cinco quick evals consecutivas abaixo de 50%, preservando a medição do nível 2. |

## Etapa 3 — limites numéricos propostos

1. No próximo diagnóstico, registrar episódio, passos, retorno total e decomposição, nível, resultado, `stuck`, distância máxima de recuo, primeira escada, plataforma e término; manter treino/teste separados.
2. Fazer quick eval de 50 episódios no treino e 50 no teste do beco a cada 250.000 passos novos. São avaliações, não passos adicionais do learner.
3. Se o teste do beco ficar abaixo de 25% em três avaliações consecutivas, apresentar a opção A para decisão; não alterar a campanha automaticamente.
4. Só propor a opção B se os registros mostrarem um modo de falha dominante ligado a progresso e o shaping correspondente não duplicar `UniversalReward`.
5. Só propor a opção C se os contadores comprovarem menos de 25% dos passos no nível 9 durante um intervalo de pelo menos 250.000 passos. Aplicar por um bloco fixo de 250.000 passos e reavaliar; não prolongar sem nova decisão.
6. Só propor a opção D após cinco quick evals consecutivas do nível 2 abaixo de 50%; manter o nível 2 em no mínimo 5% dos passos e reavaliar após 250.000 passos.
7. Interromper a hipótese de intervenção se qualquer outro nível cair mais de 5 pontos percentuais na mesma avaliação de teste; não desativar níveis nem trocar o checkpoint-base.

Esses limites são uma proposta para decisão do operador. Como o pré-treino r6 terminou, os dados independentes podem ser coletados sem reabrir a campanha. **Nenhuma campanha, mudança de reward, currículo, política ou solver foi iniciada/aplicada a partir deste diagnóstico.** A frase específica `autorizo o pré-treino 87z` ainda não foi recebida.

## Validação e próximos dados

- A validação mínima do coletor de memória e seu snapshot atual estão no relatório 87y.
- Sem séries de quick eval, retornos episódicos, fases de falha e passos por nível, a causa comportamental exata não pode ser escolhida pelos números atuais. A recomendação é instrumentar essas métricas e levar A como primeira hipótese offline para decisão, não declarar que já é a correção.
- O checkpoint íntegro permanece `universal-012940288` no runtime r6. Não retomar o treino sem a frase de autorização específica.
