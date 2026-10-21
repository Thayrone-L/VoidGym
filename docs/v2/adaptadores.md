# Adaptadores de jogo

Um adaptador traduz a interface de um jogo para o contrato comum da v2. A regra de ouro é:
`signals()` e `training_labels()` nunca entram na entrada da política.

## Ações universais

O contrato da v2 contém intenções como `LEFT`, `RIGHT`, `JUMP_TAP`, `JUMP_HOLD`, `JUMP_MAX`,
`ATTACK`, `CAST`, `DASH`, `INTERACT` e navegação de menu. A tabela completa, inclusive combinações
e durações, está em `src/voidgym/core/actions.py`. O adaptador declara os nomes na ordem usada pela
política do jogo e converte índices nos dois sentidos.

## Hollow Knight

`HollowKnightAdapter` envolve um `ExplorationTask` v1 existente. Mantém a lista atual de 43 ações,
chama `task.step`, delega a máscara de ações e expõe a última imagem. Os sinais ficam em `info`; os
rótulos são IGNORE até o prompt 86. Imports temporários da v1 têm comentário `V1-SHIM`.

## Criar um adaptador para outro jogo

1. Achar um jeito de ler a tela.
2. Achar um jeito de apertar botões.
3. Montar a tabela de ações do jogo e mapear para as universais.
4. Implementar a máscara de ações.
5. Expor sinais para recompensa.
6. Expor gabaritos opcionais (por exemplo, colisões para mapa semântico).
7. Informar escala de tempo, se o jogo permitir.
8. Testar ida e volta das ações.

Possíveis próximos adaptadores: Silksong (Unity/BepInEx), Nine Sols (Unity) e Celeste (Everest).
