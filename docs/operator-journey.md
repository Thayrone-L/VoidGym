# Jornada da campanha

O diário da jornada, o índice e o painel são telemetria do operador. A política recebe apenas `pixels` e `action_history`. Coordenadas, cena, zona, eventos e capturas ficam fora da observação; a versão `explore_v4` usa apenas os eventos confirmados como sinais de recompensa.

## Flags confirmadas pelo bridge

O bridge acompanha transições `false → true` de campos e propriedades booleanas do `PlayerData` cujo nome começa com `has` ou `got`, ou contém `boss`, `defeated` ou `killed`. O evento mantém o nome original da flag como chave. As flags que a instalação efetivamente expõe aparecem no diário e no relatório; campos sem valor inicial conhecido só passam a gerar evento depois de observada a transição. `grubsCollected` é comparado com a amostra anterior e cada aumento gera uma chave sequencial `grub_N`.

As flags com `boss`, `defeated` ou `killed` são classificadas como chefe. As outras flags aceitas são classificadas como item/ability. O acompanhamento da sombra usa a transição de `soulLimited`; morte, cena, zona, posição do Cavaleiro, posição da câmera, limites de colisores da cena e geo atual são exportados como números/texto do jogo.

## Limites e privacidade

- Posições são amostradas no máximo a cada 500 ms; o diário tem teto de 2 MiB e retenção de 5.000 runs, com compressão das mais antigas.
- Capturas de marcos usam `require_png_frame`, ficam em `runs/<campanha>/marcos`, e o limite da campanha é 32 imagens ou 24 MiB.
- `explore_v4` limita os novos marcos a 20 pontos por episódio. O máximo positivo contratual passa de 55,2 para 75,2; v1–v3 mantêm seus contratos e pesos.
- O mapa de fundo só pode ser importado/baixado por ação do operador e fica sob `%LOCALAPPDATA%/VoidLab/cache/map-art/<sha256>`. O cache nunca é copiado para relatórios ou artefatos compartilháveis. O painel mantém o esquema próprio quando não há cache ou posicionamento confiável.
- O download resolve o botão atual da página estável do MediaFire, usa `User-Agent: VoidLabJourney/1.0`, valida PNG/dimensões e registra o hash inicial. Se falhar, o arquivo pode ser importado manualmente.

## Calibração

`fit_affine` ajusta seis parâmetros com pelo menos três pares de pontos não colineares. `fit_two_point_calibration` implementa a correção manual com duas âncoras, usando escala uniforme, rotação e translação. Transformações sem dados/confiança não são inferidas pela política nem expõem cenas ainda não visitadas.

## Compartilhamento

O relatório inclui o mapa esquemático, marcos, recordes e a recomendação de avisar o autor e pedir permissão antes de distribuir o projeto. Ele não incorpora a arte do mapa.
