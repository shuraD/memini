# Memini

Generic graph memory on Neo4j, written from scratch:
- **core**: Memini's memory engine — entity graph, hierarchical attributed communities, layered index and retrieval (ideas credited where used, e.g. arXiv 2502.09891);
- **extensions**: capabilities built on the core, each with its proof of use.

> Status: **foundation**. The pipeline, ledgers and agent rules are in place; the modules follow the capability map.

## Rules
Every contributor follows **[AGENTS.md](AGENTS.md)**: people and coding agents alike (Claude Code, Agy, OpenCode). The git hooks and the CI workflow enforce it.

## Setup
```
uv sync                                 # Python 3.12 + dev tools in .venv
git config core.hooksPath .githooks     # the gate for every commit
```

## Every commit
```
.venv/bin/python -m tools.ledger change --author <who> --module <module> --summary "<what and why>"
git commit   # message: subject + Requirement:/Implemented:/Why: lines + Change-Author: trailer
```
The `pre-commit` hook runs these checks:
- format, lint and strict types;
- unit tests (≥ 90% coverage, including every bug's regression test);
- architecture contracts;
- ledger guards.

The `commit-msg` hook checks the author trailer, and that no capability is lost without `OWNER-APPROVED-REMOVAL`.

## Full pipeline (before merge)
```
tools/ci/act.sh                      # the GitHub Actions workflow, run locally in Docker (works in worktrees)
```
It adds security checks (bandit, pip-audit), dependency checks (deptry), the acceptance benchmark (Q01) and an AI review report.

## Ledger
`ledgers/*.sql` holds the features, bugs and changes, as SQL text committed with the code.
```
.venv/bin/python -c "from tools import ledger; db = ledger.load(); print(db.execute('select * from change').fetchall())"
```

## Layout
| Path | Holds |
|---|---|
| `src/memini/infra` | providers and shared infrastructure |
| `src/memini/core` | the memory engine |
| `src/memini/extensions` | capabilities built on the core |
| `config/prompts/` | every prompt (never in code) |
| `ledgers/` | the SQL ledger |
| `tools/` | ledger and CI tools |
| `tests/unit` | fast tests |
| `tests/acceptance` | benchmark questions with evidence |
