"""Benchmark eval harness: suites, config matrices, runner, reports."""

from __future__ import annotations

from kalinov.eval.compare import compare_runs
from kalinov.eval.experiment import ExperimentSpec, load_experiment
from kalinov.eval.matrix import ConfigMatrix, EvalConfig
from kalinov.eval.metrics import AggregateMetrics, aggregate
from kalinov.eval.report import build_report_payload, render_json, render_markdown, write_report
from kalinov.eval.runner import EvalRunner, RunResult, TaskResult
from kalinov.eval.suite import Suite, SuiteError, Task, TaskExpected, load_suite

__all__ = [
    "AggregateMetrics",
    "ConfigMatrix",
    "EvalConfig",
    "EvalRunner",
    "ExperimentSpec",
    "RunResult",
    "Suite",
    "SuiteError",
    "Task",
    "TaskExpected",
    "TaskResult",
    "aggregate",
    "build_report_payload",
    "compare_runs",
    "load_experiment",
    "load_suite",
    "render_json",
    "render_markdown",
    "write_report",
]
