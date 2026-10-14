# Contrato técnico v1

Este documento fixa o que precisa permanecer estável entre a ponte C# e o controlador Python. A serialização exata poderá ser JSON no spike e binária depois, mas a semântica não muda sem incrementar `protocol_version`.

## 1. Identidade

Toda mensagem inclui:

| Campo | Tipo | Regra |
|---|---|---|
| `protocol_version` | inteiro | incompatibilidade encerra handshake |
| `run_id` | UUID | uma execução do supervisor |
| `instance_id` | string | único na execução |
| `session_id` | UUID | muda a cada processo/conexão |
| `episode_id` | UUID | muda a cada reset aceito |
| `game_profile` | string | versão/categoria operacional imutável na execução |
| `seq` | uint64 | monotônico por direção |
| `sent_monotonic_ns` | uint64 | prazo, não horário civil |
| `trace_id` | UUID | correlação ponta a ponta |

Horário UTC é registrado para leitura humana; timeouts usam relógio monotônico.

## 2. Handshake

Ponte envia `HELLO`:

```json
{
  "bridge_version": "semver",
  "game_version": "string observada",
  "game_build_hash": "sha256",
  "game_profile": "development",
  "modding_api_version": "string observada",
  "pid": 0,
  "capabilities": {
    "capture": ["sync_readback", "async_gpu_readback"],
    "input_backend": ["candidate-name"],
    "events": ["player_damage", "player_death", "save_loaded"],
    "reset": ["process"]
  }
}
```

Supervisor responde `WELCOME` apenas após validar token, versões e configuração. O agente não vê `HELLO`.

O handshake falha se versão, hash, API ou conjunto de mods não corresponderem ao `game_profile`. Isso evita treinar acidentalmente em uma versão e avaliar em outra.

## 3. Comandos

- `PING`: mede disponibilidade; sem efeito no jogo.
- `CONFIGURE_CAPTURE`: resolução, formato, taxa máxima e política de drop.
- `APPLY_ACTION`: estado de botões e lease.
- `RELEASE_ALL`: prioritário, idempotente.
- `CAPTURE_ONCE`: frame diagnóstico.
- `BEGIN_EPISODE`: associa novo episódio após reset concluído.
- `PREPARE_RESET`: para captura e libera entradas.
- `SHUTDOWN`: encerramento controlado.

`APPLY_ACTION` contém:

```json
{
  "episode_id": "uuid-do-episodio-ativo",
  "action_id": 42,
  "buttons": ["LEFT", "JUMP"],
  "lease_ms": 80,
  "deadline_monotonic_ns": 0
}
```

`BEGIN_EPISODE` deve ser confirmado antes da primeira ação treinável. A ponte
rejeita `APPLY_ACTION` sem episódio ativo ou com UUID diferente. Capturas
diagnósticas continuam possíveis fora de episódio, mas não são experiência de
treino.

Comando após `deadline` é rejeitado. `lease_ms` nunca é infinito.

## 4. Respostas e eventos

- `ACK`: comando aceito e efeito agendado.
- `ACTION_APPLIED`: botões normalizados, início e fim medidos.
- `FRAME`: metadados e payload ou referência de slot.
- `EVENT`: evento permitido.
- `HEARTBEAT`: estado operacional e contadores.
- `ERROR`: código estável, recuperabilidade e texto diagnóstico.

Metadados de `FRAME`:

```json
{
  "episode_id": "uuid-do-episodio-ativo",
  "frame_id": 101,
  "capture_monotonic_ns": 0,
  "width": 320,
  "height": 180,
  "channels": 3,
  "pixel_format": "RGB8",
  "encoding": "raw|png|jpeg|shared_memory",
  "scene_transition": false,
  "player_dead": false,
  "death_persisted": false,
  "shade_pending": false,
  "shade_recovered": false,
  "run_failed": false,
  "enemy_kills_since_last_frame": 0,
  "hp_lost_since_last_frame": 0,
  "geo_gained_since_last_frame": 0,
  "geo_spent_since_last_frame": 0,
  "duplicate_of": null,
  "last_applied_action_id": 42
}
```

## 5. Eventos autorizados no protótipo

| Evento | Observação | Recompensa | `info`/operação |
|---|---:|---:|---:|
| `player_damage` | não, salvo efeito visível | somente configuração explícita | sim |
| `enemy_death_confirmed` | não | somente configuração explícita e limitada | sim |
| `player_death` | término e imagem | permitido como terminal | sim |
| `save_loaded` | não | não | sim |
| `reset_complete` | nova imagem | não | sim |
| `capture_failed` | não | não | truncamento |
| `connection_lost` | não | não | truncamento |

Eventos não autorizados são descartados na fronteira do controlador. O esquema usa uma allowlist fechada.

## 6. Dados proibidos no canal de aprendizagem

- posição/velocidade interna do personagem;
- nome ou índice interno de cena/sala;
- posição, vida ou IA de inimigos;
- inventário/progresso oculto lido do save;
- colisores, navmesh, objetos inativos ou saídas fora da câmera;
- flags de história;
- informação sobre futuros itens, chefes ou finais;
- qualquer anotação retirada de wiki, mapa, vídeo ou modelo pré-treinado.

Se um desses dados for temporariamente necessário para depuração, a execução recebe `privileged_diagnostics=true`, é marcada como inelegível para treino e seus dados ficam em diretório separado.

## 7. API Gymnasium

```python
observation, info = env.reset(seed=seed, options=options)
observation, reward, terminated, truncated, info = env.step(action)
env.close()
```

- `terminated`: morte ou sucesso real da tarefa.
- `truncated`: limite de tempo, timeout, frame inválido, desconexão, reset/falha operacional.
- `info`: visão filtrada; telemetria privilegiada nunca é repassada ao treinador genérico.
- `reset`: só retorna após frame válido pertencente ao novo `episode_id`.

O ambiente deve passar `gymnasium.utils.env_checker.check_env`, além de testes próprios de processo real.

## 8. Log de transição

Uma linha JSONL por transição aceita:

```json
{
  "schema_version": 1,
  "run_id": "...",
  "instance_id": "hk-00",
  "episode_id": "...",
  "transition_id": 0,
  "obs_before": {"frame_ids": [1, 2, 3, 4]},
  "requested_action": {},
  "applied_action": {},
  "obs_after": {"frame_ids": [2, 3, 4, 5]},
  "reward": {"total": 0.0, "components": {}},
  "terminated": false,
  "truncated": false,
  "termination_reason": null,
  "timing_ms": {},
  "valid": true
}
```

Transições rejeitadas vão para log operacional separado e nunca para rollout/replay buffer.

## 9. Estrutura de dados

```text
runs/<run_id>/
  manifest.json
  config.resolved.yaml
  environment.txt
  instances/<instance_id>/
    bridge.jsonl
    resources.csv
    episodes/<episode_id>/
      transitions.jsonl
      events.jsonl
      frames/          amostras, não necessariamente todos
      summary.json
  checkpoints/
  evaluation/
  crash/
```

Política padrão de armazenamento:

- transições e métricas: completas;
- frames: buffer de treino próprio e amostras persistidas;
- vídeo: avaliações e janelas em torno de falhas;
- rotação/limite de disco: obrigatório antes de sessões longas.

## 10. Falhas injetadas obrigatórias

- matar o controlador com botão mantido;
- bloquear heartbeat;
- atrasar e reordenar mensagens;
- duplicar comando;
- corromper payload de frame;
- desconectar durante troca de cena;
- encerrar o jogo durante escrita experimental;
- encher o limite configurado de armazenamento;
- iniciar com versão ou token incorreto;
- tentar apontar save experimental para caminho não permitido.

Todas devem resultar em estado conhecido, log legível e nenhuma alteração dos saves pessoais.
