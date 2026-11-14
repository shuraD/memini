# Architecture

Memini is generic graph memory on Neo4j. It is written from scratch, with a clean layering. All project context lives in this repository: `AGENTS.md` (rules), `docs/` (architecture and decisions), `ledgers/` (features, bugs, changes) and the specs.

## Layers (enforced by `import-linter`, `pyproject.toml`)
```
memini.extensions   capabilities built on the core, each with a proof of use
      ↑
memini.core         the memory engine: entity graph, hierarchical attributed communities,
                    layered index, layered retrieval
      ↑
memini.infra        providers (LLM, embeddings), configuration, shared plumbing
```
A lower layer never imports a higher one. The core never imports an extension.

## Principles
1. **Generic:** no business names, corpus vocabulary or fixed pipelines in `src/`. Vocabularies are data per scope.
2. **Each fact stored once:** no field duplicated by another field or by a relationship.
3. **Prompts are files:** `config/prompts/*.txt`, never strings in code.
4. **Proof of use:** every stored property and every capability has a written reader and a reason (ledger).
5. **Performance is correctness:** retrieval ≤ 100 ms p95; full answer ≤ 10 s; no model call per list item.
6. **Observable:** every question can be traced (what was retrieved, what the model received, timings per step).

## Engine (core), in build order
1. **Ingest units:** any format → units with line ranges; format detected by content.
2. **Document authority:** copies, versions, the definitive document; concept ownership per domain.
3. **Knowledge graph:** entities and relations extracted per unit, consolidated across documents.
4. **Hierarchical attributed communities:** KNN augmentation (K = average degree), weighted Leiden (weight = similarity), summaries per layer, community vectors from the summaries, layered links (M per layer, one link down).
5. **Layered retrieval:** top-down search collecting k results per layer, relations among bottom-layer hits, generation.

Ideas credited to the literature (e.g. arXiv 2502.09891) are cited in the code where used.

## Decisions
Architecture decisions are recorded as ADRs in the project's private documentation (one file per decision, never rewritten; superseded by a new one).
