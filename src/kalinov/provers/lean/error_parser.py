"""Map Lean compiler stdout/stderr into :class:`StructuredError` rows."""

from __future__ import annotations

import re

from kalinov.provers.errors import StructuredError

# Lean diagnostic starter: path:line:col: severity: message
_DIAG_LINE = re.compile(
    r"^(?P<file>[^:]+):(?P<line>\d+):(?P<col>\d+):\s*"
    r"(?P<sev>error|warning|information|note)\s*:\s*(?P<msg>.*)$",
)


def _is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    return bool(
        s.startswith(("✔", "⚠", "Building ", "Downloading ", "trace:")),
    )


def parse_lean_output(raw: str) -> tuple[StructuredError, ...]:
    """Parse Lean compiler output into structured errors."""
    out: list[StructuredError] = []
    current_msg: str | None = None
    current_meta: dict[str, str | int | None] | None = None

    def flush() -> None:
        nonlocal current_msg, current_meta
        if current_meta is None or current_msg is None:
            return
        out.append(
            StructuredError(
                severity=str(current_meta["severity"]),
                message=current_msg.rstrip(),
                file=str(current_meta["file"]) if current_meta["file"] else None,
                line=int(current_meta["line"]) if current_meta["line"] is not None else None,
                column=int(current_meta["col"]) if current_meta["col"] is not None else None,
                code="lean_diagnostic",
            ),
        )
        current_msg = None
        current_meta = None

    for line in raw.splitlines():
        m = _DIAG_LINE.match(line.rstrip("\n"))
        if m:
            flush()
            current_meta = {
                "file": m.group("file"),
                "line": m.group("line"),
                "col": m.group("col"),
                "severity": m.group("sev"),
            }
            current_msg = m.group("msg")
            continue
        if current_msg is not None and current_meta is not None:
            if _is_noise_line(line):
                continue
            if line.startswith(" ") or line.startswith("\t"):
                current_msg += "\n" + line.strip()

    flush()
    return tuple(out)
