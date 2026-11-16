# AGENTS.md — rules for every contributor (Claude Code, Agy, OpenCode, humans)

These rules are enforced by the git hooks and the CI workflow. A commit that breaks one is rejected.
Never bypass the hooks: no `--no-verify`, no changing `core.hooksPath`, no editing the hooks to pass.

## What Memini is
Generic graph memory on Neo4j, written **from scratch, clean**. Two layers, enforced by import contracts (`pyproject.toml` → `[tool.importlinter]`):
- `memini.core`: Memini's memory engine (entity graph, hierarchical attributed communities, layered index and retrieval). It never imports an extension. Where an idea comes from the literature (e.g. arXiv 2502.09891), the code cites it; the implementation is Memini's own.
- `memini.extensions`: every capability built on the core, each with its proof of use (owner decision or cited literature).
- `memini.infra`: providers and shared plumbing, below both.

No code is copied from earlier projects. Earlier work is at most a *reference* for a capability's proof; the implementation here is new.

## Owner rules (non-negotiable)
1. **Verify before building.** Fetch and confirm every external doc, API or paper before relying on it. Never invent sources.
2. **Generic, no hardcoding.** No business names, corpus vocabularies or fixed pipelines in `src/`. Corpus words may appear only in test data. **Owner 2026-11-16:**
   - No string literal of a graph label, relationship type, property or allowed value in `src/`. Code reaches them only through the schema's typed names.
   - Allowed values, keys and conditional requirements live in `config/graph_schema.yaml` (`values:`, `required_when:`), and are enforced generically for every property.
   - Every other list of values (kinds, file types, categories, roles, thresholds) lives in `config/`, never in code.
3. **Prompts live in text files** (`config/prompts/*.txt`), never in code.
4. **Never delete history.** Use status transitions (superseded), not deletes. Destructive operations need the owner's explicit approval.
5. **Never remove, switch off, or silently reshape a capability.** The feature ledger guards this. A removal needs `OWNER-APPROVED-REMOVAL: <ref>` in the commit message, and only the owner gives that ref.
6. **Prove decisions with evidence:** paper, owner decision, code `file:line`, or a measurement. Ask the owner only when the evidence conflicts or is missing.
7. **Neo4j: local Docker only.** Production is never touched. The `is_local_uri` guard refuses any host that is not this machine (it is an allowlist, so it never needs the production address). The production host is never written in the repo; it exists only in your environment as `MEMINI_PROD_NEO4J_HOST` (see `.env.example`).
8. **Never read `.env*`, `~/.ssh`, `~/.aws` or `secrets/`.** Keys come from the environment.
9. **Never change a test to make it pass.** Fix the code. A wrong expectation is reported with evidence. The gates are checks too: a change to any file in `tools/ci/protected.txt` needs `CHECK-CHANGE: <reason>` **and** `OWNER-APPROVED-CHECK-CHANGE: <the owner's words, date>`. When a gate blocks you, stop and ask the owner; never write that line without the owner's approval in your session.
10. **Requirements come from the domain responsible for them.** For example, requirements come from Product. A concept that appears in another domain is a reference, and it is flagged on conflict.
11. **Performance is part of correctness.**
    - Retrieval ≤ 100 ms p95.
    - Full answer ≤ 10 s.
    - Each fact is stored once.
    - No model call per list item.
12. **Simple architecture, no over-engineering (owner 2026-11-16).**
    - Build only what your card line asks for. A card line is a capability of the old memini, in the baseline or the version being built.
    - No protection or "enterprise" layers unless a card line names them: no access control, audit log, health score, confirmation strings or session guards.
    - No new abstraction, wrapper or option without a second real caller.
    - **Use a library instead of re-implementing it (owner 2026-11-16).** Neo4j and its GDS library first (KNN, Leiden, vector indexes, Cypher), then any maintained package. Before adding any capability (a cache, retries, JSON repair, HTTP calls, file parsing, ...), check which library already does it: less code, fewer bugs ("less code less bugs"). The spec names the library. Never hand-write what Neo4j, GDS or a library already does. Keep the old memini's file layout: a few files per job, not one file per function.
    - The old memini's logic may guide the design where it is correct; its code is never copied.
    - A slice that passes merges to `main` behind its module's flag (`src/memini/infra/flags.py`), with its ledger rows in the same commit.

## How every session works (owner 2026-11-16)
Every team session (dev1-dev4, QA, DevOps) and every orchestrator agent:
1. **Start the moment a task arrives.** Drop older work unless the owner says to keep it.
2. **Do the whole task list, then test once, then report once.** Don't stop between tasks and don't wait for confirmation of a step that is already in your list.
3. **Run only the affected tests:** the files you changed and their tests. The full suite runs only at the end, when everything is merged.
4. **No live checks unless the task asks for one,** such as starting containers or databases. The tests are the check.
5. **Read only what you need:** the files the task names, then the code you change.
6. **Fix your own mistakes and keep going.** Report them only if they change the result.
7. **Count your own time; don't blame others for it.**
8. **One short report at the end:** the commits, the files with +/- lines, the tests run with pass and fail counts, the start and end time, the cost, and only real open decisions, one line each.
9. **Log every state change** (worktrees, containers, queues, commits) as one line in `ops-ledger.jsonl`, in the orchestrator's state folder. Never push, never merge, never use `--no-verify`. vp-rnd moves main forward and the owner pushes.

**Who owns what** (nobody changes another team's files):
- **dev1:** `src/memini/infra/llm_providers/`
- **dev2:** `src/memini/core/ingest/`
- **dev3:** `src/memini/core/query/`
- **dev4:** `src/memini/infra/graph/`, `src/memini/core/archrag/`
- **DevOps:** `deploy/`, `config/`, `tools/orchestrator/`, CI and hooks
- **QA:** specs and tests
- **vp-rnd:** task lists and specs only

## How to work (TDD, small commits)
1. Read the module spec in the capability map and the relevant ledger rows.
2. **Tests first. New tests may be written by the implementer; existing tests are frozen** (owner 2026-11-14, replacing the 2026-11-12 rule).
   - **New tests:** the tests come first, from the spec, the ledger rows and the review findings, never from the implementation. The implementer may write the NEW tests for its own change, or have the test-writer role write them (private repo `agents/roles/`, installed at `~/.claude/agents/`). Then the implementer writes the minimal code until those tests pass, and refactors.
   - **Existing tests are frozen.** The agent that fixes or updates the code never changes an existing test. A failing test is a finding: fix the code, never the test. A test that is wrong goes back with evidence. Changing an existing test needs `OWNER-APPROVED-TEST-CHANGE: <the owner's words, date>` in the commit, given by the owner in the committing session.
   - **Reviews stay independent:** the reviewer is never the author of the change.
   - Name who wrote the tests in the commit: `Tests-Author: <who>`.
3. Run the checks locally: `.githooks/pre-commit`.
4. Record the change. Every commit adds exactly one ledger row:
   ```
   .venv/bin/python -m tools.ledger change --author <Claude|Agy|OpenCode|name> --module <module-id> \
       --summary "<what and why>" [--features F-001,F-002] [--bugs B-003]
   ```
   - For a new capability: add a `feature` row plus its `feature_test` rows in `ledgers/features.sql`.
   - For a fixed bug: add or update the `bug` row in `ledgers/bugs.sql`, with its regression test.
5. Commit with a message that states the decision and its reason (owner rule):
   ```
   <subject>

   Requirement: <spec/ledger/owner decision this implements>
   Implemented: <what was built — files, functions, tests>
   Why: <the reasoning; for a decision, the evidence>

   Change-Author: <who>
   Tests-Author: <who wrote the new tests; may be the Change-Author (owner 2026-11-14)>
   ```
   The hook rejects a message without all three lines and the trailer.
6. Each module is built in its own worktree: `git worktree add ../memini-<module> -b <module>`. It merges to `main` only with a green pipeline (`act`), including the acceptance benchmark.
7. **New dependencies (owner rule, 2026-11-15).** Prefer a maintained library over writing infrastructure. A new third-party package is accepted **without the owner** when all three checks pass, and each check's evidence goes in the commit message under `Why:`:
   - **Established:** its first release on PyPI is at least 1 year old.
   - **No known vulnerability:** `pip-audit` finds none in the chosen version or in any package it pulls in.
   - **Licence:** an OSI-approved licence compatible with AGPL-3.0-or-later. The planner checks the licence (PyPI classifiers) and the age; there is no automated gate for them yet (planned: F-foundation-030).
   If any check fails, or the result is unclear, the dependency is a Tier 2 owner question. A coding agent in an orchestrated slice never edits `pyproject.toml` or `uv.lock` itself: it prints `FOLLOW-UP: dependency <name>: <reason>`, and the planner checks the package and puts it in the slice spec.

## Infra and CI bugs found while working (owner 2026-11-12)
If you find a bug in the infra, CI or gates while working on your module, **do not fix it in your branch**. Record it as an open bug row with its evidence, report it, and continue your task. Infra bugs are fixed by another agent on its own branch; otherwise no task ever finishes.

## Where the context lives (all inside the repo)
- `AGENTS.md`: the rules (this file). `CLAUDE.md` points Claude Code here; other tools are pointed here by their private role configuration. Engine-specific settings (agent roles, a tool's settings file) are private local files: never tracked here, installed per user or copied into each worktree from the private source.
- `docs/ARCHITECTURE.md`: layers, principles, the engine's build order.
- Architecture decisions (ADRs): private, in the private docs repo under `adr/` (checked out at `docs/internal/adr/`); never rewritten.
- `docs/internal/specs/` (private): the capability map and module specs.
- `ledgers/`: features, bugs, changes (SQL).
- `reviews/`: the review report of each gated change.

## Checks every commit must pass
- format (`ruff format --check`)
- lint (`ruff check`)
- types (`mypy --strict`)
- **tests by tier (gate, owner 2026-11-12):** while you work, **every commit runs `tests/sanity` only** (one pass + one fail case per rule and per module, < 3 s). **Before merge to `main`** CI runs the full suite: `tests/sanity`, `tests/unit` (product code), `tests/gates` (the gate attack suites) with ≥ 90% coverage, and `tests/acceptance`. Every new module adds its pass/fail pair to `tests/sanity`.
- **no full runs while fixing (owner 2026-11-15):** after each change or fix, run only `tests/sanity` plus the test files that cover what you changed — never the whole suite locally, not even "to be safe". The full suite runs once, at the end, when everything is merged (CI). Report a local result as "green on the Mac (subset)", never as "CI green".
- architecture contracts (`lint-imports`)
- ledger guards (`tools/ci/check_ledgers.py`):
  - every active feature has an existing test;
  - every fixed bug has an existing regression test in `tests/sanity`, `tests/unit` or `tests/gates`, which runs before every merge;
  - no feature is lost without owner approval;
  - exactly one change row.

**Before merge (`tools/ci/act.sh`, the full pipeline):** the above, plus security (`bandit`, `pip-audit`), dependencies (`deptry`), the acceptance benchmark and an AI review report (`reviews/`).

## Input and benchmark
- **Corpus:** `MEMINI_CORPUS`, set in your environment (`.env`, never committed; names in `.env.example`). Read-only: never modify or copy it.
- **Benchmark:** Q01 (`tests/acceptance/questions.yaml`). The expectations are data with evidence. Each run writes a full trace and a step profile.
