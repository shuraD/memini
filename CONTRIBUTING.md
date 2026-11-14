# Contributing

All contributors (people and coding agents) follow [AGENTS.md](AGENTS.md). In short:
1. `uv sync` and `git config core.hooksPath .githooks`.
2. Work in a worktree per module: `git worktree add ../memini-<module> -b <module>`.
3. Test first, then code, then refactor.
4. Record the change: `.venv/bin/python -m tools.ledger change --author <who> --module <module> --summary "..."`.
5. Commit with a `Change-Author: <who>` trailer; the hooks run the gate.
6. Before merging: `tools/ci/act.sh` must be green, and a review report is saved under `reviews/`.
