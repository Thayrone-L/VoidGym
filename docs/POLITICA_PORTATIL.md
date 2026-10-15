# Política portátil

O VoidGym separa dois tipos de estado de treinamento:

- checkpoints SB3 (`.zip`) ficam locais e servem para retomar exatamente uma sessão,
  incluindo otimizador e contadores;
- políticas portáteis (`.safetensors`) contêm somente os tensores da política e podem
  iniciar uma nova sessão compatível em outra máquina.

Um artefato portátil sempre vem acompanhado de um manifesto fechado
`*.policy.manifest.json`. Antes de aplicar os pesos, o runtime confere SHA-256,
tamanho, versão, classe da política, fingerprint de compatibilidade, nomes, formatos,
dimensões e finitude de todos os tensores. Qualquer diferença encerra a importação sem
alterar a política existente.

O carregamento distribuído não usa `PPO.load`: uma instância PPO nova é criada, apenas
seus pesos de política são substituídos e `num_timesteps` começa em zero. Estado do
otimizador, RNG e objetos Python serializados nunca atravessam essa fronteira.

Cada checkpoint produzido por `train_ppo` recebe um par em `policies/`, e o inventário
fica em `portable_policies` no manifesto da sessão. A rotação remove os pares portáteis
na mesma cadência dos checkpoints locais.

O preflight do worker congela os dois arquivos, seus hashes e o ID do manifesto. Na
interface, uma campanha de treino pode informar opcionalmente apenas o ID dessa sessão;
o servidor resolve os caminhos canônicos no cache e volta a validar o par. Caminhos
arbitrários não fazem parte do contrato do painel.

Ao finalizar uma sessão, o outbox recebe somente `policy.safetensors` e
`policy.manifest.json`. O `checkpoint.zip` não cruza a fronteira distribuída.
