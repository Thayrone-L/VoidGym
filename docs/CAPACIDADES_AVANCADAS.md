# Capacidades avançadas em preparação

## Ações temporais

O espaço de ações distingue botões e duração. Além de movimento, pulo e ferrão,
ele contém magia normal (`CAST`), magia para cima e para baixo, cura sustentada
(`FOCUS_HOLD`) e variantes preparatórias de Nail Arts. Cura e carregamento mantêm
o botão pressionado por 1.500 ms e então liberam todas as entradas; o watchdog
limita qualquer lease a 2.000 ms.

Essas ações estão implementadas e auditáveis, mas cura, magias direcionais e Nail
Arts ainda precisam de smoke test dentro do jogo. O custo temporal da recompensa é
proporcional à duração da ação, para que canalizar não custe o mesmo que um tap.

## Memória topológica de rotas

`TopologicalRouteMemory` constrói um grafo somente com descritores visuais e eventos
conservadores de mudança estável de cenário. Uma revisita aumenta a evidência do nó
ou da aresta; nunca é bloqueada ou penalizada. Cada aresta guarda a menor sequência
de ações temporais que já realizou a travessia, e `route_plan()` devolve os segmentos,
a confiança e o número de observações.

O mapa é um artefato externo à PPO nesta etapa. Ele registra e calcula rotas, mas
ainda não assume o controle da política. Isso evita esconder estado mutável fora da
observação e transformar o processo de treino em um POMDP não declarado. Não são
usados nomes internos de salas, posição do personagem, mapa de wiki ou dados do save.

## Planejador de builds de amuletos

`CharmBuildPlanner` é a base do planejador de loadouts. Ele recebe identificadores
visuais opacos, custo em entalhes observado na UI, capacidade observada e resultados
de runs. A busca beam respeita a capacidade, considera estimativas individuais,
sinergias de pares e incerteza para exploração, sem enumerar todo o conjunto potência.

O catálogo não contém nomes ou efeitos escritos à mão. Efeitos e sinergias serão
inferidos empiricamente, comparando avaliações repetidas com uma política de gameplay
congelada. A integração de bancada/menu ainda não está implementada: ela exigirá uma
tarefa visual separada com `LEFT/RIGHT/UP/DOWN/SUBMIT/CANCEL`, confirmação visual da
mudança e recuperação segura. Equipar um amuleto por si só não gera recompensa.
