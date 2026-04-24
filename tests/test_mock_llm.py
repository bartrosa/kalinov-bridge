import pytest

from kalinov_bridge.mock_llm import fill_proof


def test_fill_proof_replaces_first_sorry() -> None:
    src = "theorem t : True := by sorry\n-- by sorry\n"
    out = fill_proof(src)
    assert "by trivial" in out
    assert out.count("by sorry") == 1


def test_fill_proof_requires_sorry() -> None:
    with pytest.raises(ValueError, match="by sorry"):
        fill_proof("theorem t : True := by trivial")
