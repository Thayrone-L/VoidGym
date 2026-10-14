# Roteiro do primeiro dia no computador com o jogo

Este roteiro para no diagnóstico e no primeiro smoke test. Não inicia treinamento.

## Informações mínimas do usuário

1. Caminho de cada instalação autônoma que poderá ser usada e a versão que se acredita existir em cada uma.
2. Confirmação de `imagem + instrumentação restrita` conforme o contrato.

Também será perguntado se há saves pessoais ou sincronização em nuvem ativa. Isso é uma verificação de segurança, não uma escolha de arquitetura.

## Etapa A — inventário somente leitura

Coletar e salvar em relatório:

- edição/build do Windows;
- CPU, RAM física e espaço livre;
- GPU, VRAM, driver NVIDIA e estado CUDA utilizável pelo PyTorch;
- Python, `pip`, .NET SDK e runtimes;
- distribuição, caminho real e versão/hash de cada `hollow_knight.exe` autônomo;
- versão das assemblies Unity e `Assembly-CSharp.dll`;
- resolução, modo de tela e limite de FPS atual;
- diretório oficial de saves e todos os arquivos presentes;
- qualquer sincronização em nuvem ou backup automático que alcance a pasta de saves;
- processos/overlays que podem interferir.

Saída: `diagnostics-<timestamp>.json` e um resumo Markdown. Nenhuma ferramenta é instalada nessa etapa.

## Etapa B — proteção de dados

1. Fechar jogo e plataforma quando necessário.
2. Inventariar tamanho, horário e SHA-256 dos saves pessoais.
3. Criar backup versionado e validar os hashes da cópia.
4. Definir uma raiz experimental explícita.
5. Impedir que configuração aceite a raiz pessoal como experimental.
6. Controlar sincronização em nuvem antes de qualquer restauração.
7. Criar um save experimental novo, sem reutilizar progresso pessoal.

Critério: um teste manual de save altera apenas o conjunto esperado; o inventário pessoal permanece idêntico.

## Etapa C — portão de compatibilidade do modding

Em 2026-10-09 existe um relato aberto de incompatibilidade entre o jogo `1.5.12459` e a Modding API. Como já existem builds autônomas e o projeto aceita múltiplas versões, isso deixa de ser bloqueio e passa a ser uma decisão de perfil. A proposta inicial é usar `1.5.78.11833` para desenvolvimento, desde que uma das builds fornecidas seja íntegra e aceite uma API compatível. Portanto:

1. inventariar e calcular o hash de todas as builds fornecidas, sem executá-las;
2. selecionar uma instalação separada para o perfil `development`;
3. conferir release/commit compatível da API e Lumafly;
4. testar com apenas a API;
5. registrar hash do jogo, API, mods, logs de inicialização e saída;
6. só depois adicionar nossa ponte mínima.

Rotas possíveis:

- `1.5.78.11833` + API compatível funcionam: fixar como `development` e continuar.
- API funciona na versão atual e oferece vantagem mensurável: podemos optar por `current`, mantendo hash exato.
- Há correção oficial/branch confiável: fixar commit/hash e testar.
- API não funciona: usar outro perfil de versão compatível; o usuário já autorizou versões antigas, mas toda instalação será separada e auditada.
- A build impede modding ou isolamento: avaliar outra build e, só depois, captura externa como fallback.

Não selecionar `1.2.2.1` como ambiente principal apenas por ser comum em speedruns. Ele será reservado para um perfil futuro de velocidade, pois o melhor ambiente de engenharia e o ambiente competitivo não precisam ser iguais.

Mesmo que os executáveis estejam em pastas independentes, assumir inicialmente que todas as builds usam o mesmo `Application.persistentDataPath` em `AppData\LocalLow`. Essa hipótese só é removida após rastrear as gravações. Cópia de pasta não conta como isolamento de save.

## Etapa D — scaffold reprodutível

Criar a estrutura do código, ambiente Python isolado e projetos C#. Fixar dependências depois de verificar compatibilidade, não a partir de números antecipados.

Checks mínimos:

- build limpo da ponte;
- testes Python sem o jogo;
- esquemas e configuração validados;
- segredos/tokens e `runs/` ignorados pelo Git;
- manifesto contém hashes e versões.

## Etapa E — smoke test da ponte

Implementar apenas:

- carregamento do mod;
- `HELLO/WELCOME`;
- heartbeat;
- `PING`;
- `RELEASE_ALL`;
- watchdog;
- encerramento limpo.

Executar testes de matar Python, fechar socket e encerrar jogo. Nenhum botão pode permanecer ativo.

## Etapa F — captura

1. capturar manualmente 100 frames com carimbos;
2. verificar HUD, orientação, cores, proporção e transições;
3. medir latência e impacto de captura síncrona;
4. testar `AsyncGPUReadback` se suportado;
5. escolher taxa/formato provisórios;
6. salvar uma folha de contato para revisão humana.

Reprovar se o frame tiver overlays de diagnóstico invisíveis ao jogador normal, recorte indevido ou envelhecimento silencioso.

## Etapa G — entrada

Para cada botão básico:

- toque curto;
- retenção;
- liberação explícita;
- expiração do lease;
- combinação válida;
- combinação conflitante;
- execução sem foco;
- desconexão durante retenção.

Registrar vídeo curto, ação solicitada/aplicada e tempos. Reprovar se depender do foco ou interferir em outra instância/processo.

## Etapa H — primeiro reset

Começar pelo reset mais conservador: encerramento controlado do processo, restauração do save experimental somente após o processo terminar e reinicialização.

Só otimizar para reset interno depois de 20 ciclos corretos. Não sobrescrever arquivo aberto. Em cada ciclo conferir hashes pessoais e arquivos experimentais alterados.

## Etapa I — ambiente completo sem aprendizagem

Executar uma política aleatória segura por 30 minutos com recompensa zero:

- validar `reset/step/close`;
- medir frames/ações/transições válidas;
- provocar falhas;
- confirmar rotação de logs;
- executar `check_env`;
- produzir relatório de aceite.

## Resultado esperado

Ao fim, uma das duas conclusões deve estar sustentada por evidência:

1. `APROVADO`: uma instância cumpre o ciclo e pode avançar à tarefa motora curta.
2. `BLOQUEADO COM CAUSA`: versão, input, captura ou saves falharam; há dados suficientes para escolher a alternativa sem arriscar progresso pessoal.

“O mod abriu” não é critério de sucesso.
