# ADR-0001b — CNL-agnostic step body

## Status

Accepted

## Context

Gherkin step bodies may embed mathematics and controlled natural language (CNL)
in several notations (LaTeX-style fragments, ForTheL, future Lean tactic syntax).
The PR 1 parser intentionally treats step text as opaque strings so the AST
stays stable while interpretation strategies evolve.

## Decision

- Step bodies remain opaque at parse time; structural Gherkin is the only
  concern of `kalinov.gherkin`.
- Semantic extraction is delegated to pluggable `StepInterpreter`
  implementations that run after parsing.
- This repository ships two initial interpreters:
  - **MathTex** — structures inline/display LaTeX-style math fragments without
    validating full LaTeX.
  - **ForTheL** — recognizes ForTheL-shaped content and optionally forwards it
    to a locally installed Naproche binary; if the tool is absent, steps are
    explicitly marked as skipped rather than failing the pipeline.
- Additional interpreters (for example Lean-specific tactics) can be added
  without changing the parser or Gherkin AST.

## Consequences

- **Positive:** Additive evolution; oracle and evaluation layers consume a
  uniform `InterpretedStep` stream regardless of surface syntax.
- **Negative:** Each interpreter defines its own `payload` schema; callers must
  branch on `interpreter_name` / `kind` rather than a single closed union type.
  We accept this in exchange for flexibility and minimal coupling to any one
  mathematical formalism.
