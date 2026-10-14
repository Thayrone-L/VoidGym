# VoidGym

Ambiente local e experimental para treinar um agente a jogar Hollow Knight a partir da própria experiência. O projeto combina uma ponte C#, um ambiente Gymnasium, treinamento PPO, campanhas contínuas, isolamento de saves e um painel web local.

O jogo, saves, Modding API, assemblies, capturas e checkpoints não fazem parte deste repositório. Cada colaborador precisa fornecer localmente uma cópia legítima compatível do jogo.

VoidGym é um projeto comunitário não oficial, sem afiliação, endosso ou patrocínio da Team Cherry. Hollow Knight e suas marcas pertencem aos respectivos titulares.

## Documentos

- [Plano mestre](docs/PLANO_MESTRE.md): escopo, arquitetura, fases, critérios de aceite e estratégia de aprendizado.
- [Contrato técnico v1](docs/CONTRATO_TECNICO_V1.md): fronteiras de informação, protocolo, estados, ações, observações, eventos e logs.
- [Roteiro do primeiro dia](docs/ROTEIRO_PRIMEIRO_DIA.md): diagnóstico e sequência de execução quando estivermos no computador com o jogo.
- [Riscos e decisões](docs/RISCOS_E_DECISOES.md): riscos, mitigação, portões e decisões ainda condicionais.
- [Ferramentas instaladas](docs/FERRAMENTAS_INSTALADAS.md): versões e hashes das ferramentas preparadas localmente.
- [Inventário de builds](docs/INVENTARIO_BUILDS.md): hashes, compatibilidade e decisão por versão do jogo.
- [Isolamento de saves](docs/ISOLAMENTO_DE_SAVES.md): launcher transacional, recuperação e limites operacionais.
- [Process owner e baseline](docs/PROCESS_OWNER_E_BASELINE.md): ciclo seguro por episódio e política aleatória reproduzível.
- [Primeira rodada de treinamento](docs/PRIMEIRO_TREINO.md): PPO diagnóstico, calibração, resultados e limites.
- [Currículo de mudança visual](docs/REACH_SCENE_CHANGE_V1.md): detector conservador, recompensa e execução segura.
- [Painel local de treinamento](docs/PAINEL_DE_TREINAMENTO.md): acompanhamento de runs, campanhas versionadas e lançamentos protegidos.
- [Capacidades avançadas](docs/CAPACIDADES_AVANCADAS.md): magia, cura, ataques carregados, memória de rotas e planejamento empírico de builds.
- [Conhecimento distribuído](docs/CONHECIMENTO_DISTRIBUIDO.md): protocolo para compartilhar métricas e linhagem entre máquinas sem publicar dados locais.

## Estado atual

`TREINAMENTO CONTÍNUO EM DESENVOLVIMENTO` — handshake, watchdog, input interno, captura, bootstrap visual, recompensas de combate/economia, memória de rotas, campanhas contínuas e isolamento de saves foram validados. O painel local pode iniciar e interromper campanhas, inclusive instâncias paralelas com saves separados.

## Decisões já firmes

1. O agente verá imagens normais da própria partida, histórico das próprias ações e o objetivo selecionado.
2. A ponte C# poderá coletar telemetria privilegiada apenas para operação e diagnóstico. Cada uso em recompensa precisará ser explícito e versionado.
3. O primeiro protótipo não usará LLM, VLM, encoder pré-treinado, demonstrações ou conhecimento específico do jogo.
4. O controle será interno ao processo e não dependerá do foco do Windows.
5. Instâncias paralelas usam processos e raízes de save independentes.
6. Os saves pessoais nunca serão usados como saves experimentais.
7. Falha, timeout ou desconexão sempre liberam todas as entradas.
8. A versão do jogo é um perfil do experimento, não uma escolha única para todo o projeto.

## Treinamento distribuído

Cada máquina treina um candidato independente a partir do campeão atual. O GitHub armazena código, métricas, hashes e linhagem; checkpoints promovidos devem ficar em Releases ou armazenamento de artefatos, nunca no histórico Git. Pesos PPO não são mesclados automaticamente.

Antes de uma rodada, importe o registro validado com `hk-agent knowledge sync`. Depois da rodada, use `hk-agent knowledge export` para produzir uma contribuição sanitizada. Consulte o documento de conhecimento distribuído para o fluxo completo e suas limitações de segurança.
