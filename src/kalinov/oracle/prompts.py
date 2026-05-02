"""Model-agnostic prompt templates for the oracle loop."""

from __future__ import annotations

import re

from kalinov.provers.base import ProofObligation
from kalinov.provers.errors import StructuredError

SYSTEM_PROMPT_LEAN: str = """\
You are a formal proof assistant for Lean 4. Given a mathematical
obligation, produce a Lean 4 source file containing exactly one theorem
that proves the obligation. Use Mathlib lemmas where helpful. Respond
with Lean 4 source only, no surrounding prose, no markdown fences.
"""

PROPOSE_TEMPLATE: str = """\
Obligation name: {name}
Statement (informal): {statement}
Hypotheses: {hypotheses_block}

Please produce a Lean 4 theorem that establishes this obligation.
"""

REPAIR_TEMPLATE: str = (
    "Your previous attempt failed.\n\n"
    "Previous Lean source:\n"
    "```lean\n"
    "{previous_body}\n"
    "```\n\n"
    "Lean diagnostics:\n"
    "{diagnostics_block}\n\n"
    "Produce a corrected Lean 4 source. Reply with Lean code only.\n"
)


def format_obligation(obl: ProofObligation) -> str:
    """Format obligation fields for user prompts."""
    if not obl.hypotheses:
        hypotheses_block = "(none)"
    else:
        hypotheses_block = "\n".join(f"- {h}" for h in obl.hypotheses)
    return PROPOSE_TEMPLATE.format(
        name=obl.name,
        statement=obl.statement,
        hypotheses_block=hypotheses_block,
    )


def body_from_llm_text(text: str) -> str:
    """Strip optional markdown fences from model output."""
    t = text.strip()
    m = re.match(
        r"^```(?:\w+)?\s*\n(.*?)\n```\s*$",
        t,
        flags=re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return t


def format_diagnostics(diags: tuple[StructuredError, ...]) -> str:
    """Serialize structured diagnostics; errors listed before warnings."""
    errors = [d for d in diags if d.severity == "error"]
    warnings = [d for d in diags if d.severity != "error"]
    lines: list[str] = []
    for d in errors + warnings:
        lines.append(f"[{d.severity}] {d.message}")
    return "\n".join(lines)


__all__ = [
    "PROPOSE_TEMPLATE",
    "REPAIR_TEMPLATE",
    "SYSTEM_PROMPT_LEAN",
    "body_from_llm_text",
    "format_diagnostics",
    "format_obligation",
]
