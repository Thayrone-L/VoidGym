# Decisão: VoidGym vira uma "IA gamer" reaproveitável entre jogos (2026-10-21)

## Objetivo (do usuário)

Depois de zerar Hollow Knight, e talvez platinar ou fazer speedrun, levar o mesmo experimento para outros jogos sem recomeçar do zero:

- Silksong;
- Nine Sols;
- Celeste.

A IA deve olhar para um jogo novo, ver algo e concluir "talvez isso seja uma plataforma, vou tentar subir". Ou seja: **senso comum de jogador**, aprendido pela **função** das coisas e não pela aparência.

## Arquitetura escolhida

1. **Contrato de adaptador de jogo (prompt 83)**
   - O núcleo da IA não conhece o Hollow Knight.
   - Cada jogo tem um adaptador que fornece:
     - tela;
     - aplicação de **ações universais**, com duração em tempo de jogo;
     - sinais opcionais para recompensa e para gabarito de treino.
   - O HK vira o primeiro adaptador, sem mudar o comportamento atual.
2. **Ginásio de plataformas (prompt 84)**
   - Mini jogo 2D procedural, rodando em CPU muito mais rápido que o HK.
   - Gera salas com plataformas, vãos, bordas altas, buracos, espinhos, inimigos simples, paredes quebráveis, portas que pedem habilidade e salas conectadas.
   - **Randomização de domínio:** formas, texturas, paletas, fundos, iluminação, câmera, tamanho do personagem e física (dentro de faixas) mudam a cada sala.
   - Só as regras ficam fixas. Por isso a IA aprende o conceito e não a arte.
   - Gabarito perfeito e gratuito: o mapa semântico de cada quadro (sólido, perigo, inimigo, quebrável, saída…).
3. **Camada de percepção "o que estou vendo" (prompt 85)**
   - O encoder visual universal prevê um mapa semântico grosseiro da tela.
   - A decisão (LSTM + cabeça de ações universais) usa esse entendimento.
   - Pré-treino no ginásio, com PPO + perdas auxiliares: semântica, dinâmica inversa e movimento do herói.
   - Avaliação em estilos nunca vistos.
4. **Transferência para o HK (prompt 86)**
   - O mod lê as colisões do HK e gera o mesmo mapa semântico **só como gabarito de treino**.
   - Uma campanha nova parte do modelo universal pré-treinado.
   - Comparação com a campanha atual: tempo para sair da Passagem do Rei e progresso por hora.
5. **Depois:** biblioteca de habilidades reutilizáveis (subir borda, atravessar vão, pular na parede, recuar depois de bater) e adaptadores para outros jogos:
   - Silksong e Nine Sols: Unity, caminho parecido com o atual;
   - Celeste: ecossistema Everest.

## Princípios que continuam valendo

- A **política** só vê a tela, o histórico de ações e o conhecimento de jogador.
- Informação interna do jogo entra só como gabarito de perdas auxiliares ou no crítico.
- **Nada específico do HK** no ginásio nem na base de senso comum.
- Save pessoal e baseline intocados.
- PPO na CPU.
- Validar pelos botões reais do painel quando houver campanha.

## Expectativa realista

- Um jogo novo ainda vai precisar de ajuste fino, mas bem mais curto.
- O básico (plataforma, perigo, inimigo, segredo) já vem pronto.
- O primeiro teste de que a ideia funciona: a IA pré-treinada sai da Passagem do Rei claramente mais rápido que a atual.

## Precedentes

- **SIMA** (DeepMind): agente em vários jogos 3D.
- **Procgen** (OpenAI): generalização com níveis procedurais.
- **Randomização de domínio:** robótica, do simulador para o mundo real.
- **Aprendizado de affordances.**

## Ideia em avaliação: especialistas (exploração × combate/boss), 2026-10-21

A proposta do usuário: um modelo especializado em exploração e inimigos simples, e outro em bosses. Ao encontrar um boss, o primeiro passa o controle ao segundo.

**Avaliação:** é uma boa arquitetura, desde que os especialistas **compartilhem o "olho" e a memória** em vez de serem dois modelos totalmente separados.

```
tela → olho universal (percepção) → LSTM ─┬─ especialista EXPLORAÇÃO (ator)
                                          ├─ especialista COMBATE/BOSS (ator)
                                          └─ seletor (contexto: "estou numa luta de boss?")
```

**Vantagens:**

- Treinar contra bosses não estraga a exploração, e o contrário também vale. Isso evita o "esquecimento catastrófico".
- A prática de desafios (63) e as demonstrações de boss (79) treinam só o especialista de combate.
- O especialista de combate pode ser pré-treinado num modo **arena** do ginásio, com chefes procedurais: padrões de ataque com aviso visual, fases, arena fechada. Essa variedade é universal e não é cópia de nenhum boss real.

**Seletor:** no começo, por regra de contexto, usando o que o jogador percebe: a arena fechou, há um inimigo grande com barra ou padrão de chefe. Depois, ele pode ser aprendido.

**Cuidados:**

- Transição suave na troca de especialista, com o mesmo estado da LSTM.
- Dados divididos entre os dois.
- O crítico pode ser compartilhado ou separado; a escolha vai ser medida.

**Quando:** depois do 87, como prompt 88 (arena do ginásio + especialista de combate). Os slots de "especialistas" já devem ser previstos na política do 85 (ator com cabeças nomeadas) para não precisar refatorar depois.
