# Pendências operacionais

Este arquivo registra bloqueios externos que não devem impedir o desenvolvimento e o
treino local. Uma pendência só é removida depois de uma verificação prática.

## Publicação distribuída por fork

- **Estado:** pendente, não bloqueante para treino local.
- **Motivo:** a conta `Thayrone-L` é proprietária de `VoidGym` e não pode criar um
  fork do próprio repositório na mesma conta.
- **Necessário:** criar `<outra-conta>/VoidGym` como fork em outra conta GitHub
  confiável ou usar o fork de um colaborador.
- **Depois:** substituir `configure-me/VoidGym` em `config.worker.local.json`, adicionar
  o remote `fork` por HTTPS, autenticar somente pelo Git Credential Manager e executar
  uma publicação de teste.
- **Critério de conclusão:** `prepare-publication` e `publish-fork` produzem uma branch
  inédita no fork, o SHA remoto coincide com o recibo e a URL de comparação abre um PR
  manual contra `Thayrone-L/VoidGym:knowledge`.

Enquanto esta pendência existir, `sync`, `preflight`, treinamento e finalização local
continuam permitidos. Somente a etapa de envio ao fork permanece desabilitada.

