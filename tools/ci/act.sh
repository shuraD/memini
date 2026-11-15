#!/usr/bin/env bash
# Run the full gate pipeline locally (the same workflow GitHub would run).
#
# act copies the working directory into the container (never --bind: a bound
# directory let the container's `uv sync` rewrite the host .venv with Linux binaries).
# The repository's common git dir is mounted READ-ONLY at the same path, so the
# ledger guards read real history from a worktree too (B-007: a gate must read real
# history or fail), and the host checkout is never modified.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
common="$(cd "$(git rev-parse --git-common-dir)" && pwd)"
exec act -W .github/workflows/ci.yml \
  -P ubuntu-24.04=catthehacker/ubuntu:act-24.04 \
  -P ubuntu-latest=catthehacker/ubuntu:act-latest --rm \
  --container-options "-v ${common}:${common}:ro" "$@"
