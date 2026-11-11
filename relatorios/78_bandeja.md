# Etapa 78 — VoidGym na bandeja

## Implementação

Adicionado um cliente WinForms nativo em `tools/VoidGymTray`, sem dependências NuGet adicionais. O supervisor conserva a propriedade dos processos do treino. O cliente usa a API de 77 em 8765/8766, lê o token CSRF por `/api/session`, consulta uma vez a cada 5 s, aplica recuo quando há erro e mantém logs em `artifacts/control-plane/tray-AAAAmmdd.log`.

O menu inclui abrir painel, retomar campanha com checkpoint, reiniciar após parada segura, parar com segurança, abortar com confirmação, minimizar/mostrar janelas identificadas como descendentes dos PIDs registrados e iniciar com o Windows por atalho opcional na pasta Inicializar. Fechar o ícone não fecha o supervisor. O botão de saída oferece só sair, parar com segurança e sair, ou cancelar. Clique duplo e clique na notificação abrem o painel.

O aborto novo usa apenas os Job Objects associados ao grupo selecionado, marca os lançamentos como `ABORT_REQUESTED` e deixa o control plane capturar os saves isolados ao encerrar os runners. Se o supervisor recuperou um lançamento sem o Job Object original, recusa o aborto e explica a indisponibilidade, em vez de matar processos por nome.

## Validação

- `dotnet restore` e `dotnet build -c Release`: aprovados; 0 avisos, 0 erros.
- Self-test da bandeja: aprovado (parsing de estado, campanha selecionada, mutex de instância única, hierarquia de PIDs, filtro de janelas fora da árvore e confirmação de aborto/saída).
- Testes Python focados de aborto do grupo e relay autenticado: 2 passaram.
- Ruff e `git diff --check`: aprovados.
- O executável foi aberto na sessão do usuário e outra abertura terminou sem criar uma segunda instância (1 processo permaneceu).

## Validação real pendente

Não foi possível interagir com o menu pela automação visual desta sessão: Computer Use havia encerrado sua execução anterior porque não conseguiu confirmar o URL ativo do navegador. O acesso HTTP local por shell também está bloqueado nesta sessão. Assim, não foram confirmados o estado visual do ícone, o abrir/reabrir do painel pela bandeja, o toast no Windows, minimização/restauro e o ritmo de coleta com o jogo minimizado.

A campanha principal chegou a R154/R155 e foi encerrada pelo supervisor às 23:02:02 por `learner heartbeat expirou após 1800s`; o checkpoint R154 foi preservado. O ícone não enviou comandos de treino. A campanha não foi reiniciada, respeitando a preferência anterior do operador de iniciar o treino manualmente. Por isso, os testes de continuar/parar/abortar via menu foram substituídos por testes unitários e de API; o aborto em campanha curta permanece por validar.
