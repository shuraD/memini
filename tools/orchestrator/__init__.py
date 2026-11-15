"""Deterministic task orchestrator: the atomic slice workflow (owner 2026-11-14).

One script replaces persistent coordination sessions: dispatch one task, a stateless
one-shot QA agent writes the acceptance tests, a stateless one-shot coder implements,
deterministic gates and a one-shot diff audit decide, and one squash commit lands on main.
Entry point: ``python -m tools.orchestrator --help``.
"""
