# Painel local de treinamento — guia do operador

O painel acompanha e controla treinos do Hollow Knight nesta máquina. Ele fica disponível
somente em `http://127.0.0.1:8765`.

## Iniciar e acompanhar

1. Para só inspecionar, rode `.\.venv\scripts\python.exe tools\serve_training_dashboard.py`. Esse modo não abre jogos.
2. Para operar um treino real, rode `.\.venv\scripts\python.exe tools\serve_training_dashboard.py --allow-execute`.
3. No painel, escolha 1–3 instâncias e a duração: até parar, por horas ou até um horário. A recomendação considera capacidade configurada e CPU.
4. Leia a confirmação antes de iniciar: serão abertas janelas do jogo; feche seu jogo pessoal primeiro.
5. Acompanhe salas, avaliações, mortes, recuperações, velocidade, CPU, memória e alertas na tela. A área “Avançado para desenvolvedor” contém logs e caminhos técnicos.

## Parar ou continuar

- Use **Parar com segurança**. Aguarde **Parado com segurança ✓**, que confirma a parada cooperativa e a restauração dos saves isolados.
- Um limite de horas ou horário fica gravado nos registros dos processos. Se o painel reiniciar, o prazo continua sendo aplicado e solicita a mesma parada segura.
- **Continuar de onde parou** cria uma nova versão da campanha e usa o checkpoint mais recente do learner compartilhado.
- **Repetir treino** cria outra campanha, com nova semente e sem reutilizar o checkpoint anterior.

## Relatórios

Ao terminar uma campanha, o painel gera um HTML autocontido e um JSON em
`artifacts/reports/<campanha>/`. **Gerar relatório** também cria os arquivos sob demanda.
A lista de relatórios anteriores permite abrir o HTML ou baixar o JSON.
Os relatórios incluem evolução, salas e primeira descoberta, avaliações, mortes, sombras,
alertas, reinícios, checkpoints, checagens dos saves e motivo do encerramento.

## Observações

A pontuação é escolhida pela versão do treino e não pode ser alterada no painel. Campanhas
antigas continuam registradas com seus próprios pesos. O servidor usa apenas a biblioteca
padrão do Python, não invoca shell e escuta apenas em `127.0.0.1`.
