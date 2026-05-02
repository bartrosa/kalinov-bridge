Feature: ForTheL to Lean bridge (example)

  Optional Naproche translation into Lean is wired through `kalinov check --prover lean4`
  when steps are recognized as ForTheL claims.

  @lean
  Scenario: Tagged claim (Lean obligations + ForTheL path when enabled)
    Then [ForTheL] Theorem. Every set x satisfies x = x.
