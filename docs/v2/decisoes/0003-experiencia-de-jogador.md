# Decisão: a IA joga como um jogador normal (2026-10-19)

## Princípio (decisão do usuário)

A IA deve ter as mesmas ferramentas de um jogador humano, e nunca informação que o jogador ainda não teria naquele momento. Por exemplo, ela não sabe o que é a Garra de Louva-Deus antes de pegá-la. **Ao pegar, ela passa a saber para que serve**, como o jogo ensina o jogador nesse momento.

Objetivo de longo prazo: **um dia zerar o jogo**. Por isso, ela recebe no momento certo o que o jogo ensina (tutoriais, objetivos), para não depender 100% do acaso.

**Fronteira de informação atualizada.** A observação da política passa a ser:

- a tela;
- o histórico de ações;
- o **conhecimento de jogador**:
  - o que ela já obteve e para que serve;
  - os lugares especiais já encontrados e para que servem;
  - o objetivo atual quando o jogo o revela;
  - a memória do que ela mesma já explorou;
  - os pontos de interesse que ela mesma marcou;
  - os inimigos na tela, com a memória do que ela viveu com cada tipo.

Continua proibido: mapa ou estado do jogo que o jogador não conheceria, como salas ainda não visitadas, segredos, posição de itens não vistos, itens ainda não obtidos ou atributos internos dos inimigos.

## 1. Jogada contínua (prompt 57; substitui o reset do save a cada episódio)

- **Save persistente:** o save de cada instância evolui como numa jogada real. O que ela conquista fica.
- **Retorno pelo jogo:** a morte devolve ao último banco, pela mecânica do próprio jogo, com a sombra.
- **Checkpoints do próprio jogo:** a IA recomeça de onde parou. Hoje é a Passagem do Rei; depois, os bancos (Dirtmouth, Encruzilhada…).
- **Episódio é só uma fatia de tempo para o PPO,** não um reset. Se empacar, ela "recarrega o último banco" (Salvar e Sair → Continuar), como um jogador faria.
  - **Revisão (2026-10-20, prompt 68):** antes do primeiro banco, essa recarga devolve ao início da Passagem do Rei e apaga o progresso. Por isso "travada" passa a só fechar a fatia do episódio e ela continua de onde está; recarregar pelo menu fica só para travamento real.
- **Backups:** cópia do save a cada banco, para recuperar de corrupção ou softlock. Opção "começar jogada nova" a partir da baseline. O save pessoal continua intocável.
- **Learner compartilhado:** cada ator tem a sua própria jogada, e o learner continua compartilhado.

## 2. Conhecimento de jogador na observação (prompt 58)

- **Itens obtidos:** para cada habilidade, item-chave ou amuleto, um bloco com "obtido" e um descritor do efeito.
  - O descritor tem categoria, o que habilita, o botão usado e, para amuletos, entalhes e efeitos.
  - Fica zerado até a coleta e é preenchido a partir do que o próprio jogo informa ao jogador.
  - A observação não leva nomes, só índices e descritores.
- **Ações novas destravadas pela habilidade:** DASH, SUPER_DASH e DREAM_NAIL entram no espaço de ações, mascaradas até a habilidade ser obtida. Um pequeno bônus limitado nas primeiras vezes em que a ação nova é usada com efeito.
- **Memória de exploração própria:**
  - um mapa pessoal grosseiro da sala atual, com as células por onde ela já passou;
  - as saídas da sala, marcando quais já usou e se levam a lugar conhecido.

  Tudo isso é construído só com o que ela viveu.
- **Pontos de interesse marcados automaticamente:** onde morreu, onde ficou presa, onde viu algo que não conseguiu alcançar. É o equivalente aos pinos do jogo.

## 3. Voltar com habilidade nova

A novidade vitalícia passa a contar separadamente para cada conjunto de habilidades. Depois da garra, lugares antigos voltam a valer a pena, porque agora há caminhos novos a tentar.

## 4. Tutoriais, interação e objetivos (prompt 59)

- **Interagir (UP):** sempre disponível. Serve para sentar, falar, entrar em portas e tocar sinos.
- **Ações de menu ou diálogo:** só quando há menu ou diálogo aberto na tela.
- **Lugares especiais ensinados no 1º encontro:**
  - **Banco:** salva, vira ponto de retorno, cura e permite trocar amuletos.
  - **Estação de besouro:** viagem rápida, sino, custo em geo.
  - **Loja e cartógrafo**, entre outros.
- **Memória dos bancos e estações já vistos,** para saber voltar a eles.
- **Objetivos da história no ritmo do jogo:** tabela própria de etapas, cada uma com:
  - gatilho de revelação, no momento em que o jogador recebe a dica;
  - tipo do objetivo;
  - alvo no nível de detalhe que o jogo dá.

  Recompensa ao concluir a etapa. "Nível de dica": 0 = só o jogo (padrão); 1 = guia leve, como um amigo que já jogou.
- **Recompensas 1× por lugar:** primeiro descanso em cada banco, abrir uma estação, primeira viagem.

## 5. Bestiário pessoal (prompt 60)

- **Inimigos na tela** (só os visíveis, até K):
  - posição relativa na tela;
  - tipo como ID anônimo;
  - golpes já dados neste encontro.

  Nada de HP ou atributos internos do jogo.
- **Memória por tipo, só do vivido e persistida na jogada:**
  - encontros e vitórias;
  - golpes para derrotar;
  - dano tomado e mortes causadas;
  - geo médio e itens que já deu;
  - se ela costuma fugir ou lutar.

  Antes do 1º encontro: "nunca visto".
- **O geo vale pelo que compra:** a IA lembra os preços que viu em lojas visitadas. Comprar algo novo é recompensado; geo e abate repetido não pagam por si só. Farmar só compensa se o geo for usado depois.
- **Painel:** "Bestiário da IA" para o operador, com nomes amigáveis só na tela dele.

## 6. Amuletos no banco (prompt 61)

Gerenciar amuletos no banco: menu, entalhes e combinações.

## 7. Memória de curto prazo (prompt 62)

Política recorrente (LSTM) com máscara de ações, implementação própria (MaskableRecurrentPPO). É necessária para aprender **padrões de ataque** e sequências de vários segundos: hoje ela vê só ~0,4 s. Os estados da LSTM persistem entre fatias e são zerados só em descontinuidades reais.

## 8. Treino de desafios (prompt 63)

Decisão do usuário: tentar, morrer, aprender e tentar de novo é justo, como um jogador comum faz.

- **Desafio** é um chefe, um trecho difícil do cenário ou um item/área avistado e não alcançado.
- **Snapshot de treino** na entrada do desafio. Atores de treino repetem o desafio e alimentam o mesmo learner.
- **A jogada principal** ainda precisa vencer o desafio "de verdade".
- **Memória por desafio:** tentativas, melhor progresso e vitórias. Reencontros com versões mais difíceis aproveitam a base.

## 9. Pedir ajuda (prompt 64; revisado em 2026-10-20, prompt 69)

Decisão do usuário: gravações humanas não são o padrão; são recurso para dificuldade grande, e **a IA pede**.

- **Gatilho:** por exemplo, 100 tentativas sem sucesso num desafio, ou horas sem progresso.
- **Pedido no painel:** "Pode mostrar como você faz?".
- **Gravação:** a pessoa joga uma cópia a partir do snapshot; o save pessoal não é tocado.
- **Uso pelo learner:** um termo de imitação restrito àquele desafio, com peso que decai até zero.
- **Colaboradores:** amigos poderão gravar no mesmo formato.

**Revisão de 2026-10-20 (decisão do usuário).** Não há orçamento para milhões de tentativas num problema básico; após ~62 mil passos ela ainda não sai do buraco do início nem sobe as plataformas da Passagem do Rei. Então a ajuda entra **cedo**:

- **Gatilho padrão:** travar **N = 10 vezes no mesmo lugar** (cena + célula da grade da sala), ajustável no painel, ou X minutos sem bater o recorde de progresso da sala. Limite de pedidos por hora. Se ninguém responder, ela continua jogando.
- **Demonstração ao vivo:** o operador assume aquela instância; os botões são convertidos para a ação mais próxima do espaço de ações. A outra instância segue treinando.
- **Aprender o geral, não o ponto:** o banco de demonstrações é persistente e marcado por tipo (subir borda, sair de buraco, plataforma). A perda de imitação entra em todas as atualizações seguintes. Os dados são aumentados por espelhamento (com troca de esquerda/direita) e por variações visuais, e não se usa a posição do lugar como entrada.
- **Prática logo depois:** algumas tentativas a partir do ponto em que ela travava, dentro da regra justa do treino de desafios; depois ela volta ao ponto onde o humano parou.
- **Medir a generalização:** tempo para passar em lugares parecidos que não receberam ajuda.
- **Checagem prévia:** confirmar que o pulo mais longo do espaço de ações (JUMP_HOLD ≈ 160 ms hoje) alcança essas bordas. Se não alcançar, nenhuma ajuda resolve.

## Depois

Comprar mapa e usar o mapa e os pinos do jogo.
