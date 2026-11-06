# Etapa 80 — Marcadores da DLL e painel resiliente

## Incidente e recuperação

O painel respondia 504 em `/api/operator`. O status identificou o grupo `jogada-1-repeat-1c3b-v0018` com learner `FAILED` por heartbeat expirado e nenhum ator. Solicitei a parada pela rota autenticada `/api/operator/stop`; ela retornou `STOP_REQUESTED`, o status passou a `active_count=0` e o processo do learner encerrou. Não houve encerramento forçado.

Antes da implantação, comparei o artefato Release da etapa 79 com as duas cópias instaladas. O SHA-256 era idêntico nas três: `4b81745094ecd5d984f702d1b587333c7e417eff6fc66c9aaf20a59bb0748057`. Os dois marcadores guardavam o hash antigo `5510d89d0c09a947f69662b8925c09e291f2c05d8cea3c4683d3c6867ce1b460`.

## Implementação

Adicionei `tools/deploy_bridge.py` e `refresh_instance_bridge_marker` em `instance_install.py`. O comando cria backup da DLL e marcador, copia o build, confirma o hash, atualiza o marcador atomicamente pela rotina oficial e chama `load_instance_marker` para validar. Se uma validação falha, restaura os arquivos anteriores da instância. A regra ficou documentada em `tools/README.md`.

`operator_snapshot` agora isola a parte de jogadas contínuas: uma falha produz listas vazias e um aviso em vez de derrubar o snapshot. Marcador de instância inválido vira o alerta vermelho **“A instância 1 está com a DLL diferente do registro — rode a implantação da DLL”**. Como fallback, `_do_get` retorna HTTP 200 com um snapshot renderizável mesmo se a montagem geral falhar.

O deploy em `hk-ai-1` e `hk-ai-2` foi concluído por `tools/deploy_bridge.py`; ambos foram validados com `load_instance_marker`. Backups: `C:\VoidGym\artifacts\bridge-deploy-backups\bridge-20261021-000617\hk-ai-1` e `...\hk-ai-2`.

## Validação

- Testes focados: 49 passaram antes do teste específico do comando de deploy.
- Suíte completa: **697 passaram, 2 ignorados, 3 avisos conhecidos do Stable-Baselines, 8 subtestes**, em 134,57 s.
- `ruff check src tests tools\deploy_bridge.py`: passou.
- `mypy src`: passou (55 arquivos).
- `git diff --check`: passou; Git apenas avisou sobre conversão CRLF→LF em arquivos já CRLF.
- Self-deploy real: hash idêntico ao build 79 e marcadores válidos para as duas instâncias.

## Validação operacional pendente

O painel deve ser recarregado para importar as alterações da etapa 80. A continuação agora aceita o número de atores do seletor; sem esse campo, clientes antigos preservam a contagem da campanha anterior. A tentativa da rodada 19 foi cancelada durante a inicialização e terminou com `active_count=0`; o manifesto registra `final_save_status=captured`. Conforme instrução do operador, nenhuma campanha nova será iniciada hoje. Por isso, não foi possível validar `/api/operator` abaixo de 300 ms com campanha ativa nem executar o soak de 15 minutos; esses dois critérios operacionais ficam explicitamente pendentes. A compilação e as validações automatizadas já passaram (699 passaram, 2 ignorados, 3 avisos conhecidos, 8 subtestes; Ruff, mypy e `git diff --check` aprovados).
