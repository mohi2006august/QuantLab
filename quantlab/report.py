"""Self-contained HTML reports for measured run results."""

import html
from pathlib import Path
from typing import Any


def pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    measured = [row for row in rows if row.get("status") == "completed"]
    frontier = []
    for candidate in measured:
        candidate_ppl = candidate.get("perplexity")
        candidate_size = candidate.get("size_gib")
        candidate_speed = candidate.get("tok_s")
        dominated = False
        if candidate_ppl is None or candidate_size is None or candidate_speed is None:
            frontier.append(candidate)
            continue
        for other in measured:
            if other is candidate or any(other.get(metric) is None for metric in ("perplexity", "size_gib", "tok_s")):
                continue
            no_worse = (
                other["perplexity"] <= candidate_ppl
                and other["size_gib"] <= candidate_size
                and other["tok_s"] >= candidate_speed
            )
            strictly_better = (
                other["perplexity"] < candidate_ppl
                or other["size_gib"] < candidate_size
                or other["tok_s"] > candidate_speed
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def write_html_report(run_id: str, rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frontier = pareto_frontier(rows)
    frontier_keys = {(row["method"], row["variant"]) for row in frontier}
    table_rows = []
    for row in rows:
        is_pareto = (row["method"], row["variant"]) in frontier_keys
        values = [row.get("method"), row.get("variant"), row.get("status"), row.get("perplexity"), row.get("custom_score"), row.get("task_scores_json"), row.get("size_gib"), row.get("batch_size"), row.get("kv_cache_dtype"), row.get("tok_s"), row.get("tok_s_std"), row.get("prefill_ms_estimate"), row.get("ttft_ms"), row.get("error")]
        cells = "".join(f"<td>{html.escape(_display(value))}</td>" for value in values)
        table_rows.append(f'<tr class="pareto">{cells}<td>{"yes" if is_pareto else ""}</td></tr>')

    numeric = [row for row in rows if row.get("size_gib") is not None and row.get("perplexity") is not None]
    svg = _scatter_svg(numeric, frontier_keys)
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QuantLab run {html.escape(run_id)}</title><style>
body{{font:15px system-ui,sans-serif;background:#0f1117;color:#e5e7eb;margin:0;padding:32px}}main{{max-width:1200px;margin:auto}}h1{{font-size:28px}}.muted{{color:#9ca3af}}section{{background:#171a22;border:1px solid #292e38;border-radius:10px;padding:20px;margin:20px 0;overflow:auto}}table{{border-collapse:collapse;width:100%;white-space:nowrap}}th,td{{padding:10px 12px;border-bottom:1px solid #303642;text-align:left}}th{{color:#a3e635}}.pareto td:last-child{{color:#a3e635}}svg{{max-width:100%;height:auto}}.note{{font-size:13px;color:#9ca3af}}
</style></head><body><main><h1>QuantLab results</h1><p class="muted">Run {html.escape(run_id)} · quality, artifact size and throughput</p>
<section><h2>Pareto view</h2>{svg}<p class="note">Lower perplexity and artifact size are better. Table below includes exact measurements.</p></section>
<section><h2>Run matrix</h2><table><thead><tr><th>Method</th><th>Format</th><th>Status</th><th>Perplexity</th><th>Custom score</th><th>Task scores (JSON)</th><th>Size (GiB)</th><th>Batch size</th><th>KV dtype</th><th>tok/s</th><th>tok/s std dev</th><th>Prefill estimate (ms)</th><th>TTFT (ms)</th><th>Error</th><th>Pareto</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></section>
</main></body></html>"""
    output_path.write_text(document, encoding="utf-8")
    return output_path


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _scatter_svg(rows: list[dict[str, Any]], frontier_keys: set[tuple[str, str]]) -> str:
    if not rows:
        return '<p class="muted">No completed measurements to plot yet.</p>'
    width, height, margin = 720, 360, 48
    max_size = max(float(row["size_gib"]) for row in rows) or 1
    min_ppl = min(float(row["perplexity"]) for row in rows)
    max_ppl = max(float(row["perplexity"]) for row in rows)
    ppl_range = max(max_ppl - min_ppl, 0.01)
    dots = []
    for row in rows:
        x = margin + float(row["size_gib"]) / max_size * (width - margin * 2)
        y = height - margin - (float(row["perplexity"]) - min_ppl) / ppl_range * (height - margin * 2)
        key = (row.get("method", ""), row.get("variant", ""))
        color = "#a3e635" if key in frontier_keys else "#60a5fa"
        title = html.escape(f"{row.get('method')} {row.get('variant')}: {row.get('perplexity')} PPL, {row.get('size_gib')} GiB")
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}"><title>{title}</title></circle>')
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Perplexity by artifact size scatter plot"><line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#6b7280"/><line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#6b7280"/><text x="{width/2}" y="{height-8}" fill="#9ca3af" text-anchor="middle">Artifact size (GiB)</text><text x="16" y="{height/2}" fill="#9ca3af" transform="rotate(-90 16 {height/2})" text-anchor="middle">Perplexity ↓</text>{"".join(dots)}</svg>'
