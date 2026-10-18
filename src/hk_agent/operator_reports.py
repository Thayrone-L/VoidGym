from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from typing import Any


def build_report(summary: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Build the durable JSON and self-contained HTML operator report."""
    payload = dict(summary)
    payload.setdefault("schema_version", 1)
    payload.setdefault("report_type", "hollow_knight_training_run")
    payload.setdefault("sections", [
        "Resumo", "Salas descobertas", "Evolução", "Avaliações", "Mortes e sombras",
        "Alertas e reinícios", "Checkpoints e saves", "Motivo do fim",
    ])
    def table_rows(key: str, columns: Sequence[str]) -> str:
        values = payload.get(key, [])
        if not isinstance(values, list) or not values:
            return f'<tr><td colspan="{len(columns)}">Sem registros</td></tr>'
        result = []
        for row in values:
            if isinstance(row, Mapping):
                result.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(column, '—')))}</td>" for column in columns) + "</tr>")
        return "".join(result) or f'<tr><td colspan="{len(columns)}">Sem registros</td></tr>'

    scene_rows = table_rows("scenes_discovered", ("scene_name", "first_seen"))
    episode_rows = table_rows("episode_progress", ("episode", "steps", "distinct_scenes", "deaths", "shade_recoveries", "enemy_kills", "termination_reason", "return"))
    evaluation_rows = table_rows("evaluations", ("round", "steps", "return", "completed", "rollout_submitted"))
    checkpoints = payload.get("checkpoints_saved", [])
    checkpoint_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in checkpoints) if isinstance(checkpoints, list) and checkpoints else "<li>Nenhum checkpoint registrado</li>"
    alerts = payload.get("alerts", [])
    alert_items = "".join(f"<li>{html.escape(str(item))}</li>" for item in alerts) if isinstance(alerts, list) and alerts else "<li>Nenhum alerta registrado</li>"
    rows = {
        "campaign": payload.get("campaign_id", "—"),
        "duration": payload.get("duration_seconds", 0),
        "instances": payload.get("instances", 0),
        "steps": payload.get("training_steps", 0),
        "updates": payload.get("learner_updates", 0),
        "deaths": payload.get("deaths", 0),
        "shade_recoveries": payload.get("shade_recoveries", 0),
        "shade_losses": payload.get("shade_losses", 0),
        "instance_restarts": payload.get("instance_restarts", 0),
        "save_checks": payload.get("save_safety_checks", "—"),
        "end_reason": payload.get("end_reason", "—"),
    }
    summary_rows = "".join(f"<tr><th>{html.escape(key.replace('_', ' ').title())}</th><td>{html.escape(str(value))}</td></tr>" for key, value in rows.items())
    document = f'''<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Relatório {html.escape(str(payload.get("campaign_id", "treino")))}</title><style>
body{{font:16px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#17212b;background:#f4f7f8}}h1,h2{{color:#103b4d}}section{{background:white;border:1px solid #d5e0e4;border-radius:12px;padding:1rem 1.3rem;margin:1rem 0}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:.55rem;border-bottom:1px solid #e2eaed}}th{{width:28%}}small{{color:#50636e}}
</style><h1>Relatório do treino</h1><small>Campanha: {html.escape(str(payload.get("campaign_id", "—")))}</small>
<section><h2>Resumo</h2><table>{summary_rows}</table></section>
<section><h2>Salas descobertas</h2><table><thead><tr><th>Sala</th><th>Primeira descoberta</th></tr></thead><tbody>{scene_rows}</tbody></table></section>
<section><h2>Evolução por episódio</h2><table><thead><tr><th>Episódio</th><th>Passos</th><th>Salas</th><th>Mortes</th><th>Sombras recuperadas</th><th>Abates</th><th>Fim</th><th>Retorno</th></tr></thead><tbody>{episode_rows}</tbody></table></section>
<section><h2>Avaliações periódicas</h2><table><thead><tr><th>Rodada</th><th>Passos</th><th>Retorno</th><th>Concluída</th><th>Enviada ao treino</th></tr></thead><tbody>{evaluation_rows}</tbody></table></section>
<section><h2>Alertas e reinícios</h2><ul>{alert_items}</ul></section>
<section><h2>Checkpoints e segurança dos saves</h2><ul>{checkpoint_items}</ul><p>Checagem de saves: {html.escape(str(rows['save_checks']))}</p></section>
<section><h2>Motivo do fim</h2><p>{html.escape(str(rows['end_reason']))}</p></section></html>'''
    # Validate that the JSON twin can be written without NaN or non-JSON values.
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return payload, document
