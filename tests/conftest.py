"""Pytest configuration shared by LLM tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _llm_single_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid long tenacity backoffs in adapter error-path tests."""
    monkeypatch.setenv("LLM_MAX_RETRIES", "1")
