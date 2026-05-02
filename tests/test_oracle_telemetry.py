"""Tests for ``oracle_loop.jsonl`` and transcript sidecars."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from kalinov.oracle import OracleConfig, OracleLoop, OracleOutcomeKind
from kalinov.provers import NullProver, NullProverConfig, NullProverMode, ProofObligation
from kalinov.telemetry import start_run
from tests.fixtures.fake_llm_client import FakeLLMClient


def test_one_line_per_iteration(tmp_path: Path) -> None:
    llm = FakeLLMClient()
    llm.set_queue(["a", "b", "c"])
    cfg = OracleConfig(max_repair_attempts=2)
    loop = OracleLoop(
        prover=NullProver(NullProverConfig(mode=NullProverMode.ALWAYS_FAIL)),
        llm=llm,
        model="gpt-4o",
        config=cfg,
    )
    obl = ProofObligation(name="g", statement="s", hypotheses=())
    with start_run(runs_dir=tmp_path) as run:
        out = asyncio.run(loop.run(obl))
        olp = run.run_dir / "oracle_loop.jsonl"
    lines = olp.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert out.kind is OracleOutcomeKind.GAVE_UP


def test_lines_link_to_llm_and_prover_calls(tmp_path: Path) -> None:
    llm = FakeLLMClient()
    llm.set_queue(["theorem k := rfl"])
    loop = OracleLoop(
        prover=NullProver(NullProverConfig(mode=NullProverMode.ALWAYS_OK)),
        llm=llm,
        model="gpt-4o",
    )
    obl = ProofObligation(name="g2", statement="s", hypotheses=())
    with start_run(runs_dir=tmp_path) as run:
        asyncio.run(loop.run(obl))
        run_dir = run.run_dir
    llm_ids = {}
    for ln in (run_dir / "llm_calls.jsonl").read_text(encoding="utf-8").strip().splitlines():
        row = json.loads(ln)
        llm_ids[row["call_id"]] = row
    prov_ids = {}
    for ln in (run_dir / "prover_calls.jsonl").read_text(encoding="utf-8").strip().splitlines():
        row = json.loads(ln)
        prov_ids[row["call_id"]] = row
    for ln in (run_dir / "oracle_loop.jsonl").read_text(encoding="utf-8").strip().splitlines():
        ol = json.loads(ln)
        assert ol["llm_call_id"] in llm_ids
        assert ol["prover_call_id"] in prov_ids


def test_transcript_saved_when_configured(tmp_path: Path) -> None:
    llm = FakeLLMClient()
    llm.set_queue(["theorem k := rfl"])
    cfg = OracleConfig(save_transcripts=True)
    loop = OracleLoop(
        prover=NullProver(NullProverConfig(mode=NullProverMode.ALWAYS_OK)),
        llm=llm,
        model="gpt-4o",
        config=cfg,
    )
    obl = ProofObligation(name="tx/ob", statement="s", hypotheses=())
    with start_run(runs_dir=tmp_path) as run:
        asyncio.run(loop.run(obl))
        tpath = run.run_dir / "transcripts" / "tx_ob.json"
    assert tpath.is_file()
    data = json.loads(tpath.read_text(encoding="utf-8"))
    assert "messages" in data


def test_no_transcript_when_disabled(tmp_path: Path) -> None:
    llm = FakeLLMClient()
    llm.set_queue(["theorem k := rfl"])
    loop = OracleLoop(
        prover=NullProver(NullProverConfig(mode=NullProverMode.ALWAYS_OK)),
        llm=llm,
        model="gpt-4o",
    )
    obl = ProofObligation(name="g", statement="s", hypotheses=())
    with start_run(runs_dir=tmp_path) as run:
        asyncio.run(loop.run(obl))
        tdir = run.run_dir / "transcripts"
    assert not tdir.exists()
