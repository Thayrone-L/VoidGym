# Adaptadores de jogo

Um adaptador traduz observações e intenções entre um jogo e a política universal. A regra de
ouro é que `signals()` e `training_labels()` nunca entram na entrada da política: ficam para
recompensa, telemetria e supervisão auxiliar.

| Intenção | Campos principais |
|---|---|
| mover | `move` -1/0/+1 |
| mirar | `aim`: none/up/down |
| pular | `jump`: instant/short/hold/max |
| atacar / esquivar | `attack`, `dash` |
| habilidade | `special` |
| interagir / menu | `interact`, `menu` |

O shim de Hollow Knight mantém a tabela e os índices da v1, fornece quadros RGB, envia ações,
replica a máscara disponível e expõe sinais de treino separadamente. Os rótulos ainda são
`None`; o prompt 86 pode preencher esses gabaritos.

## Adaptar outro jogo

1. Descubra como ler a tela.
2. Descubra como enviar botões.
3. Declare as ações e mapeie para intenções universais.
4. Implemente a máscara de ações.
5. Exponha sinais para recompensa.
6. Exponha gabaritos opcionais, sem alimentar a política.
7. Informe a escala de tempo se disponível.
8. Teste ida e volta da tabela.

Candidatos para depois: Silksong e Nine Sols podem usar adaptadores Unity; Celeste pode usar
Everest. Nenhum deles é implementado neste prompt.
