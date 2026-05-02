"""Canonical JSON reports and Markdown summaries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from kalinov.cost.models import TokenUsage
from kalinov.eval.metrics import AggregateMetrics, aggregate, is_finite_number
from kalinov.eval.runner import RunResult, TaskResult
from kalinov.oracle.strategy import OracleAttempt, OracleOutcome
from kalinov.provers.base import ProofArtifact, ProofObligation


def _kalinov_version() -> str:
    try:
        return version("kalinov-bridge")
    except PackageNotFoundError:
        return "unknown"


def pricing_yaml_sha256() -> str:
    """SHA-256 of the bundled ``pricing.yaml`` (determinism fingerprint)."""
    here = Path(__file__).resolve().parent.parent / "cost" / "pricing.yaml"
    return hashlib.sha256(here.read_bytes()).hexdigest()


def _decimal_str(d: Decimal) -> str:
    return format(d, "f")


def _token_usage_dict(u: TokenUsage) -> dict[str, int]:
    return {
        "input": u.input,
        "output": u.output,
        "reasoning": u.reasoning,
        "cache_read": u.cache_read,
        "cache_write": u.cache_write,
    }


def _obligation_dict(o: ProofObligation) -> dict[str, Any]:
    return {
        "name": o.name,
        "statement": o.statement,
        "hypotheses": list(o.hypotheses),
        "metadata": dict(o.metadata),
    }


def _artifact_dict(a: ProofArtifact | None) -> dict[str, Any] | None:
    if a is None:
        return None
    return {
        "body": a.body,
        "language": a.language,
        "metadata": dict(a.metadata),
        "obligation": _obligation_dict(a.obligation),
    }


def _attempt_dict(a: OracleAttempt) -> dict[str, Any]:
    cost = a.cost
    return {
        "iteration": a.iteration,
        "check_result_ok": a.check_result.ok if a.check_result is not None else None,
        "cost_usd": None if cost is None else _decimal_str(cost.total_usd),
        "duration_ms": a.duration_ms,
        "artifact": _artifact_dict(a.artifact),
    }


def _outcome_dict(o: OracleOutcome) -> dict[str, Any]:
    return {
        "obligation": _obligation_dict(o.obligation),
        "kind": o.kind.value,
        "attempts": [_attempt_dict(a) for a in o.attempts],
        "final_artifact": _artifact_dict(o.final_artifact),
        "total_cost_usd": _decimal_str(o.total_cost_usd),
        "diagnostic": o.diagnostic,
    }


def _task_result_dict(tr: TaskResult) -> dict[str, Any]:
    return {
        "task_id": tr.task.id,
        "task_file": str(tr.task.file),
        "expected": tr.task.expected.value,
        "tags": list(tr.task.tags),
        "config_label": tr.config_label,
        "obligations_total": tr.obligations_total,
        "obligations_solved": tr.obligations_solved,
        "outcomes": [_outcome_dict(o) for o in tr.outcomes],
        "total_cost_usd": _decimal_str(tr.total_cost_usd),
        "total_tokens": _token_usage_dict(tr.total_tokens),
        "duration_ms": tr.duration_ms,
        "matched_expected": tr.matched_expected,
        "telemetry_run_id": tr.telemetry_run_id,
    }


def _run_result_dict(rr: RunResult) -> dict[str, Any]:
    return {
        "suite_id": rr.suite.suite_id,
        "suite_description": rr.suite.description,
        "config": {
            "label": rr.config.label,
            "prover_name": rr.config.prover_name,
            "provider_name": rr.config.provider_name,
            "model": rr.config.model,
            "seed": rr.config.seed,
            "oracle": {
                "max_repair_attempts": rr.config.oracle.max_repair_attempts,
                "max_tokens_per_call": rr.config.oracle.max_tokens_per_call,
                "temperature": rr.config.oracle.temperature,
            },
        },
        "task_results": [_task_result_dict(t) for t in rr.task_results],
        "started_at": rr.started_at.isoformat(),
        "ended_at": rr.ended_at.isoformat(),
    }


def _aggregate_dict(m: AggregateMetrics) -> dict[str, Any]:
    return {
        "config_label": m.config_label,
        "suite_id": m.suite_id,
        "tasks_total": m.tasks_total,
        "tasks_solved": m.tasks_solved,
        "tasks_gave_up": m.tasks_gave_up,
        "tasks_errored": m.tasks_errored,
        "obligations_total": m.obligations_total,
        "obligations_solved": m.obligations_solved,
        "success_rate": m.success_rate,
        "total_cost_usd": _decimal_str(m.total_cost_usd),
        "total_tokens": _token_usage_dict(m.total_tokens),
        "median_attempts_to_solve": m.median_attempts_to_solve,
        "error_breakdown": dict(m.error_breakdown),
    }


def build_report_payload(results: Sequence[RunResult]) -> dict[str, Any]:
    metrics = aggregate(results)
    return {
        "report_version": 1,
        "kalinov_version": _kalinov_version(),
        "pricing_yaml_sha256": pricing_yaml_sha256(),
        "aggregate_metrics": [_aggregate_dict(m) for m in metrics],
        "run_results": [_run_result_dict(r) for r in results],
    }


def render_json(results: Sequence[RunResult]) -> str:
    return json.dumps(build_report_payload(results), indent=2, sort_keys=True) + "\n"


def render_markdown(results: Sequence[RunResult]) -> str:
    """Human-readable tables; JSON remains canonical."""
    if not results:
        return "# Eval report\n\n(no runs)\n"

    metrics = aggregate(results)
    lines: list[str] = []
    lines.append("# Eval report\n")
    lines.append(f"- Kalinov version: `{_kalinov_version()}`")
    lines.append(f"- Pricing YAML SHA-256: `{pricing_yaml_sha256()}`\n")

    lines.append("## Aggregate metrics\n")
    lines.append(
        "| Config | Suite | Success rate | Cost (USD) | "
        "Tokens (in/out) | Median attempts (solved) | Tasks ok / gave up / err |\n",
    )
    lines.append("|---|---|---:|---:|---:|---:|---|\n")
    for m in metrics:
        sr = m.success_rate
        sr_s = f"{sr:.4f}" if is_finite_number(sr) else "n/a"
        med = m.median_attempts_to_solve
        med_s = f"{med:.2f}" if is_finite_number(med) else "n/a"
        tok = m.total_tokens
        lines.append(
            f"| `{m.config_label}` | {m.suite_id} | {sr_s} | {_decimal_str(m.total_cost_usd)} | "
            f"{tok.input}/{tok.output} | {med_s} | "
            f"{m.tasks_solved} / {m.tasks_gave_up} / {m.tasks_errored} |\n",
        )

    lines.append("\n## Per-task outcomes\n")
    config_labels = [r.config.label for r in results]
    header = "| Task | " + " | ".join(f"`{c}`" for c in config_labels) + " |\n"
    sep = "|---|" + "|".join(["---:" for _ in config_labels]) + "|\n"
    lines.append(header)
    lines.append(sep)

    task_ids = [t.task.id for t in results[0].task_results]
    by_cfg: dict[str, dict[str, TaskResult]] = {}
    for r in results:
        by_cfg[r.config.label] = {tr.task.id: tr for tr in r.task_results}

    for tid in task_ids:
        row = [tid]
        for cl in config_labels:
            tr = by_cfg[cl].get(tid)
            if tr is None:
                row.append("—")
            else:
                row.append(
                    f"{tr.obligations_solved}/{tr.obligations_total} "
                    f"({_decimal_str(tr.total_cost_usd)})",
                )
        lines.append("| " + " | ".join(row) + " |\n")

    lines.append("\n## Error breakdown (obligation outcomes)\n")
    for m in metrics:
        lines.append(f"### `{m.config_label}`\n")
        if not m.error_breakdown:
            lines.append("_none_\n")
        else:
            for k, v in sorted(m.error_breakdown.items()):
                lines.append(f"- `{k}`: {v}\n")

    lines.append("\n## Telemetry run IDs\n")
    for r in results:
        lines.append(f"### `{r.config.label}`\n")
        for tr in r.task_results:
            lines.append(
                f"- **{tr.task.id}**: `{tr.telemetry_run_id}` → `runs/{tr.telemetry_run_id}/`\n",
            )

    return "".join(lines)


def write_report(
    results: Sequence[RunResult],
    *,
    out_dir: Path,
    formats: tuple[str, ...] = ("json", "md"),
) -> Mapping[str, Path]:
    """Write ``report.json`` / ``report.md`` under *out_dir*."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    if "json" in formats:
        p = out_dir / "report.json"
        p.write_text(render_json(results), encoding="utf-8")
        written["json"] = p
    if "md" in formats:
        p = out_dir / "report.md"
        p.write_text(render_markdown(results), encoding="utf-8")
        written["md"] = p
    return written


__all__ = [
    "build_report_payload",
    "pricing_yaml_sha256",
    "render_json",
    "render_markdown",
    "write_report",
]
