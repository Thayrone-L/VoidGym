# Conhecimento de jogador

A política de exploração visual do contrato `visual_motion_player_knowledge_v1` observa três fontes: pixels capturados, histórico das ações aplicadas e conhecimento acumulado pela própria jogada. Sinais de recompensa, nomes de cenas, flags do bridge e coordenadas brutas permanecem em `info`/telemetria e nunca são incluídos diretamente na observação.

O vetor `player_knowledge` contém blocos anônimos de aquisição e descritores numéricos para habilidades, itens-chave e os 40 índices de amuletos. O bloco inteiro, inclusive o descritor, fica zerado até o bridge confirmar a aquisição. O bridge lê o custo em entalhes do próprio `PlayerData` apenas para amuletos obtidos. O vetor usa apenas categoria, código de efeito, custo e ação; os nomes próprios não saem do repositório para a política.

Categorias: `1` movimento, `2` combate, `3` cura/vida, `4` utilidade, `5` alma/magia, `6` invocação. Os códigos de efeito dos amuletos usam o índice de aquisição do jogo como chave anônima para esta tabela textual:

| Código | Efeito descrito ao jogador |
|---:|---|
| 1 | Recolhe geo solto nas proximidades |
| 2 | Indica a posição atual quando o mapa está aberto |
| 3 | Gera alma ao receber dano |
| 4 | Prolonga a invulnerabilidade após dano |
| 5 | Bloqueia dano enquanto concentra alma, até quebrar |
| 6 | Aumenta o dano quando resta pouca vida |
| 7 | Acelera a concentração para curar |
| 8 | Acrescenta duas máscaras de sangue vital ao descansar |
| 9 | Acrescenta quatro máscaras de sangue vital ao descansar |
| 10 | Emite uma nuvem que afeta inimigos próximos |
| 11 | Transforma o projétil mágico em vários projéteis menores |
| 12 | Ataca inimigos próximos após receber dano |
| 13 | Aumenta bastante o alcance do ferrão |
| 14 | Evita o recuo ao acertar com o ferrão |
| 15 | Aumenta o empurrão causado pelo ferrão |
| 16 | O dash sombrio causa dano e percorre mais distância |
| 17 | Emite esporos durante a concentração de alma |
| 18 | Aumenta o alcance do ferrão |
| 19 | Aumenta o dano de magias |
| 20 | Aumenta a alma obtida ao acertar com o ferrão |
| 21 | Aumenta bastante a alma obtida ao acertar com o ferrão |
| 22 | Consome alma para invocar filhotes de proteção |
| 23 | Acrescenta máscaras de vida |
| 24 | Aumenta geo obtido de inimigos |
| 25 | Aumenta o dano do ferrão |
| 26 | Acelera o carregamento das artes do ferrão |
| 27 | Troca vida comum por sangue vital e impede cura por concentração |
| 28 | Permite mover-se enquanto concentra alma |
| 29 | Recupera uma máscara perdida após algum tempo sem dano |
| 30 | Carrega o ferrão dos sonhos mais rápido e rende mais alma ao acertar |
| 31 | Reduz o intervalo do dash e libera dash para baixo |
| 32 | Aumenta a velocidade dos golpes do ferrão |
| 33 | Reduz o custo de alma das magias |
| 34 | Duplica a cura por concentração, que demora mais |
| 35 | Dispara um feixe pelo ferrão quando a vida está cheia |
| 36 | Gera alma gradualmente; a forma evoluída também altera a relação com o vazio |
| 37 | Aumenta a velocidade de corrida |
| 38 | Cria um escudo que acompanha o jogador |
| 39 | Invoca ajudantes que atacam e protegem |
| 40 | Invoca um companheiro que evolui e ataca inimigos |

A grade atual tem 16×9 células. Ela é preenchida somente por posições amostradas durante a própria execução e só a grade da sala atual é observada. A troca de cena registra uma saída usada na célula de transição; salas não visitadas não são enumeradas. Eventos de morte, travamento e novidade podem gerar pontos de interesse quando a própria telemetria os registra.

O fingerprint muda com a versão do contrato e a assinatura do espaço de observação. A novidade vitalícia combina a cena visual com o conjunto fixo de aquisições; aprender uma habilidade torna lugares antigos elegíveis de novo. O warm-start parcial copia somente tensores da política antiga cujos nomes e dimensões coincidam, preservando a CNN visual quando compatível; novas entradas, cabeças incompatíveis e novas saídas de ação mantêm a inicialização da política nova.

## Tutoriais, lugares e objetivos (explore_v4 schema 6)

O bridge libera `INTERACT` (UP curto) em toda situação. `MENU_UP/DOWN/LEFT/RIGHT/SUBMIT/CANCEL` só ficam disponíveis com diálogo ou menu ativo; submit/cancel usam as bindings próprias do menu. Durante diálogo/menu, ações de gameplay ficam mascaradas. O estado vem de `DialogueBox.currentConversation` e do pause/menu ativo do jogo.

A política guarda lugares especiais somente quando um componente está dentro da câmera e sua posição pode ser convertida para a grade já visitada: banco, estação de besouro, loja e cartógrafo. O descritor de cada tipo começa em zero e é preenchido no primeiro encontro; a grade inclui apenas coordenadas já vistas. Preços entram na memória só enquanto itens de loja estão ativos na tela. A lista não enumera pontos futuros.

`src/hk_agent/story_objectives.py` mantém uma sequência curta e extensível: `enteredTutorialFirstTime` revela alcançar a Encruzilhada; `metElderbug` revela derrotar o Falso Cavaleiro; `falseKnightDefeated` revela alcançar o Caminho Verde (cenas `Fungus1*`). O estágio, tipo e progresso do objetivo são códigos anônimos no vetor. A conclusão e a primeira chegada pagam uma vez por jogada, dentro do teto compartilhado de recompensa de marcos (20 por episódio). Descansar num banco paga uma vez por célula de banco observada; a abertura de estação e a primeira viagem rápida pagam uma vez por evento confirmado.

O parâmetro `tutorial_hint_level` aceita `0` (padrão: somente os gatilhos oficiais da tabela) e `1` (guia leve no vetor quando o mesmo objetivo oficial está ativo). O nível não revela objetivos futuros. A tabela usa flags de progresso do jogo que mudam durante a conversa/descoberta; não há direção codificada além do alvo que essa flag revela.

## Bestiário pessoal (explore_v4 schema 7)

Cada frame pode observar até seis inimigos dentro da câmera, ordenados pela distância ao centro. A política recebe posição normalizada, código de tipo opaco, golpes que já aplicou no encontro e nove valores de memória; os slots ausentes são zero. O bridge não transmite HP, dano do inimigo nem padrões. A memória persistida por jogada cresce a partir de aparições visíveis, dano recebido quando a atribuição é inequívoca e eventos de abate com golpes e Geo observados. A flag de item e a contagem de mortes por tipo só mudam quando a telemetria entrega esses eventos; sem sinal, continuam desconhecidas/zero. O indicador de combate estima encontros em que houve golpe antes do inimigo desaparecer, não prova intenção de fugir.

Os IDs são hashes estáveis de nomes internos normalizados e nunca incluem o nome no vetor. A tabela de nomes amigáveis é usada apenas pelo painel. Os preços permanecem limitados ao que foi visto em lojas abertas. O vetor bestiário tem seis slots de 13 valores; o espaço de observação e fingerprint mudam, com warm-start apenas dos tensores compatíveis da imagem.

A persistência pessoal está no schema 4 e o contrato de exploração no schema 8, mantendo leitura dos schemas anteriores. O escritor usa arquivo temporário exclusivo, flush/fsync e retries curtos para colisões transitórias de substituição no Windows. Um learner antigo só pode fazer warm-start dos tensores visuais cujos nomes e formas coincidirem.

## Loadouts de amuletos no banco (explore_v4 schema 8)

`OPEN_CHARMS` envia PAUSE e só fica disponível quando `at_bench` é verdadeiro e não há outro menu ou diálogo aberto. Com menu ativo, a máscara libera somente NOOP, INTERACT e os controles de menu. O bridge reporta `charm_menu_open`, `charm_cursor`, o bitmask de 40 amuletos equipados e entalhes usados/total; a observação de conhecimento guarda flags de equipados somente para itens adquiridos e acrescenta os entalhes normalizados.

`PlayerKnowledge.owned_charm_loadouts()` gera candidatos a partir de `gotCharm_N` e custos de entalhe observados no save, limitado à capacidade atual e a 256 combinações. O macro `build_charm_menu_plan` aceita somente banco sentado e página de amuletos aberta; cada pulso requer uma captura PNG calibrada e o passo final exige a imagem de gameplay. A política escolhe a combinação; o macro só navega, confirma, verifica cada frame e sai com CANCEL. Se um frame não bater, o executor libera todos os botões e aborta.

Trocar amuletos não gera recompensa. O diário guarda segmentos por sala e loadout, com frames, Geo, dano, inimigos derrotados e mortes observados; o painel os mostra como resultados ocorridos durante cada trecho, sem afirmar que o loadout causou o resultado.