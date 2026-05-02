# Corpus

This directory holds **human-facing** formalization inputs and **machine-mined** starting points.

## Layout

- **`mined/`** — default output for `kalinov mine --out …`. Files here are **candidates** only: they use generated names (`<feature>__<hash>.feature`) and the `@mined` tag.

## Attribution

Each mined scenario is preceded by `# @attribution` comment lines recording:

- **source** — which upstream adapter produced the item (e.g. `arxiv`).
- **url** — stable link to the specific document or page.
- **license** — best-effort string from the source (e.g. arXiv’s non-exclusive distribution tag); **not legal advice**.
- **retrieved_at** — UTC timestamp when the item was fetched.

The emitter refuses to write scenarios without URL and license fields it can attach.

## Review before suites

Mined `.feature` files are **not** ready for benchmark suites as-is. Before adding them under `evals/suites/` or similar:

1. Read and edit the informal claims and steps.
2. Rename to a stable, descriptive filename.
3. Remove or adjust `@mined` if the content is no longer “raw mine output”.

## License diligence

**You** are responsible for checking that your use of mined text complies with upstream licenses and institutional policies. Kalinov records metadata it can obtain from sources; it does not verify fitness for redistribution, training, or publication.
