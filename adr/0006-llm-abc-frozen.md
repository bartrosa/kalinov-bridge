# ADR-0006 — LLMClient ABC frozen

## Status

Accepted

## Context

The benchmark harness must call multiple LLM vendors without a unifying third-party
router. Each provider exposes different knobs (extended thinking, cache control,
reasoning effort, local base URLs) and different usage accounting. Downstream
code (oracle loop, cost reports, replays) needs a single stable contract and
machine-readable errors.

## Decision

- **ABC:** `LLMClient` in `kalinov.llm.base` exposes:
  - `complete(...)` — non-streaming call returning `Completion`.
  - `stream(...)` — yields `str` chunks; default implementation delegates to
    `complete` in one shot. Providers may override for true streaming.
  - `count_tokens(...)` — best-effort preflight count (heuristic for local
    OpenAI-compatible servers).
  - `capabilities(model)` — optional metadata mapping.
- **Types:** `Message` (role + text), `Completion` (text, `TokenUsage`,  
  `model_id_resolved`, `raw_response: Any`, `cache_hit`), `LLMError` and
  `BudgetExceededError` (subclass) with `code` in a small controlled vocabulary.
- **Errors:** Adapters wrap SDK/HTTP failures as `LLMError` (never leak raw SDK
  types). `raw_response` remains `Any` so we store provider-native objects for
  forensics without committing to a cross-vendor schema.
- **Provider extras:** Pass-through `Mapping[str, Any]`; semantically important keys:
  - **Anthropic:** `extended_thinking_budget_tokens` → `thinking` on
    `messages.create`; `cache_control` reserved for future block-level wiring.
  - **OpenAI:** `reasoning_effort` for reasoning models.
  - **Gemini / compat:** no required keys in 4a.
- **Cache key:** SHA-256 of canonical JSON containing `provider` registry key,
  **model alias** (not the resolved id), full message list, `max_tokens`,
  `temperature`, `stop`, and the allow-listed extras subset. Values are the
  response text, `raw_response` JSON (best-effort), `usage`, and resolved model.
  *Trade-off:* alias + params must match; resolved model is **not** part of the
  key so stable aliases keep working across API version bumps.
- **Telemetry:** `log_llm_call` appends to `runs/<id>/llm_calls.jsonl` for every
  attempt (success, cache hit, failure) with `Decimal` cost serialized as
  string USD.
- **Budget:** `BudgetGuard` (thread-safe) accumulates `CostBreakdown` and
  `TokenUsage` after each **non-cached** success; raises `BudgetExceededError`
  when any configured ceiling would be exceeded after recording the latest call.
  Binding is via `kalinov.llm.budget_context` for upcoming orchestration.

## Consequences

- New providers add a focused adapter file (~150–250 LOC) plus fixtures instead
  of touching orchestration.
- Callers must not depend on `raw_response` structure—only on `Completion` and
  `TokenUsage` fields.
- Cache replay in `READ_ONLY` mode requires identical prompts/params; otherwise
  expect `LLMError` with a logged `cache_miss_read_only` line.

## References

- ADR-0002 (native SDKs) and ADR-0003 (telemetry JSONL streams).
