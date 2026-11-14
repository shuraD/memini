# Security

- Secrets live only in the environment (`.env` is git-ignored and never read by agents).
- Neo4j: local Docker only in development and tests; production databases are never touched by tooling.
- Before every merge, the pipeline (`.github/workflows/ci.yml`, run locally with `act`) runs `bandit` on `src/` and `tools/` (medium severity and above) and `pip-audit` on the Python dependencies. The per-commit hook does not run them.
- Report a vulnerability privately to sha1sha1sha1sha1024@gmail.com; do not open a public issue.
