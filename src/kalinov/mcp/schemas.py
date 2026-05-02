"""Pydantic I/O models for MCP tools.

USD amounts are **strings** (decimal text), never floats — floats lose cents.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class SolveRequest(BaseModel):
    feature_path: str = Field(
        description="Path to a .feature file. Absolute or relative to server CWD.",
    )
    prover: Literal["null", "lean4"] = Field(
        default="null",
        description="'null' for offline testing; 'lean4' requires elan and built runtime.",
    )
    provider: str = Field(description="Provider name from kalinov.config.yaml.")
    model: str | None = Field(default=None, description="Override the provider's default model.")
    max_repair_attempts: int = Field(default=3, ge=0, le=20)
    max_cost_usd: str | None = Field(
        default=None,
        description="Hard budget cap as a decimal string (e.g. '5.00'). Not a float.",
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    save_transcripts: bool = Field(default=False)

    @field_validator("max_cost_usd", mode="before")
    @classmethod
    def _max_cost_string_not_float(cls, v: object) -> object:
        if isinstance(v, bool) or v is None:
            return v
        if isinstance(v, (int, float)):
            msg = "max_cost_usd must be a decimal string (e.g. '5.00'), not a number"
            raise ValueError(msg)
        return v


class SolveOutcomeSummary(BaseModel):
    obligation_name: str
    kind: Literal[
        "solved",
        "gave_up",
        "budget_exceeded",
        "prover_error",
        "llm_error",
    ]
    iterations: int
    total_cost_usd: str
    final_artifact: str | None
    diagnostic: str | None


class SolveResponse(BaseModel):
    ok: bool = True
    error: str | None = None
    run_id: str = ""
    runs_dir: str = ""
    outcomes: list[SolveOutcomeSummary] = Field(default_factory=list)
    total_cost_usd: str = "0"
    duration_ms: int = 0


class CheckRequest(BaseModel):
    feature_path: str
    prover: Literal["null", "lean4"] = "null"
    no_forthel: bool = False
    null_mode: Literal["always_ok", "always_fail", "fail_after_n"] = "always_ok"
    null_fail_after: int = 0


class CheckResultEntry(BaseModel):
    obligation_name: str
    ok: bool
    diagnostics: list[str]
    duration_ms: int


class CheckResponse(BaseModel):
    ok: bool = True
    error: str | None = None
    run_id: str = ""
    results: list[CheckResultEntry] = Field(default_factory=list)


class EvalRequest(BaseModel):
    suite_path: str
    prover: Literal["null", "lean4"] = "null"
    providers: list[str] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=lambda: [42])
    max_repair_attempts: int = 3
    max_cost_usd: str | None = None
    out_dir: str | None = None


class EvalResponse(BaseModel):
    ok: bool = True
    error: str | None = None
    run_ids: list[str] = Field(default_factory=list)
    report_paths: dict[str, str] = Field(default_factory=dict)
    summary_markdown: str = ""
    total_cost_usd: str = "0"


class MineRequest(BaseModel):
    source: Literal["arxiv"] = "arxiv"
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    out_dir: str = "corpus/mined"
    network: bool = Field(
        default=False,
        description="Required True for real fetch.",
    )


class MineResponse(BaseModel):
    ok: bool = True
    error: str | None = None
    run_id: str = ""
    emitted_paths: list[str] = Field(default_factory=list)
    candidate_total: int = 0


class CostReportRequest(BaseModel):
    run_id: str | None = None
    runs_dir: str | None = None
    group_by: Literal["none", "provider", "model", "day"] = "none"


class CostReportResponse(BaseModel):
    ok: bool = True
    error: str | None = None
    total_usd: str = "0"
    total_tokens: dict[str, int] = Field(default_factory=dict)
    breakdown: list[dict[str, Any]] = Field(default_factory=list)
    pricing_snapshot_sha: str = ""


__all__ = [
    "CheckRequest",
    "CheckResponse",
    "CheckResultEntry",
    "CostReportRequest",
    "CostReportResponse",
    "EvalRequest",
    "EvalResponse",
    "MineRequest",
    "MineResponse",
    "SolveOutcomeSummary",
    "SolveRequest",
    "SolveResponse",
]
