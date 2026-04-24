import json
import subprocess
from pathlib import Path

from kalinov_bridge.mock_llm import fill_proof
from kalinov_bridge.pipeline import run_demo_cycle


def test_run_demo_cycle_restores_scratch_and_records_result(tmp_path: Path) -> None:
    scratch = tmp_path / "Scratch.lean"
    scratch.write_text("theorem runner_target : True := by sorry\n", encoding="utf-8")
    lean_dir = tmp_path / "lean"
    lean_dir.mkdir()
    (lean_dir / "lakefile.toml").write_text("", encoding="utf-8")
    artifacts = tmp_path / "out"

    def fake_lake(_: Path) -> subprocess.CompletedProcess[str]:
        inner = scratch.read_text(encoding="utf-8")
        assert "by trivial" in inner
        assert "by sorry" not in inner
        return subprocess.CompletedProcess(
            args=("lake", "build"),
            returncode=0,
            stdout="",
            stderr="ok",
        )

    result = run_demo_cycle(
        scratch_file=scratch,
        lean_dir=lean_dir,
        artifacts_dir=artifacts,
        fill_proof=fill_proof,
        lake_build=fake_lake,
    )

    assert scratch.read_text(encoding="utf-8") == "theorem runner_target : True := by sorry\n"
    assert result.success is True
    assert result.returncode == 0
    payload = json.loads((artifacts / "results.jsonl").read_text(encoding="utf-8").strip())
    assert payload["task"] == "run-demo"
    assert payload["success"] is True
    assert (artifacts / "lake_stderr.txt").read_text(encoding="utf-8") == "ok"
    assert (artifacts / "Scratch.patched.lean").read_text(encoding="utf-8").count("by trivial") == 1
    assert (artifacts / "Scratch.original.lean").read_text(encoding="utf-8") == (
        "theorem runner_target : True := by sorry\n"
    )
    assert payload["patched_lean_name"] == "Scratch.patched.lean"
    assert payload["original_lean_name"] == "Scratch.original.lean"
