"""Performance summary of a run, computed from call-log rows.

Rows are grouped per provider and per role. Embedding batches form groups of their
own (their key carries the row kind) so they never distort generation percentiles.
Percentiles and throughput use real calls only; cache hits count toward ``calls`` and
the cache-hit percentage. Percentiles use the nearest-rank method.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from memini.infra.llm_providers.calllog import CallRecord
from memini.infra.llm_providers.config import LLMConfig


@dataclass(frozen=True)
class GroupStats:
    """Aggregated performance of one group of call-log rows."""

    calls: int
    cache_hit_pct: float
    p50_s: float
    p95_s: float
    max_s: float
    tokens_per_s: float
    failures_by_reason: Mapping[str, int]
    fallbacks: int
    wait_s: float
    cost: float


@dataclass(frozen=True)
class RunSummary:
    """Per-provider and per-role statistics plus role budgets as ``(used, limit)``."""

    by_provider: Mapping[str, GroupStats]
    by_role: Mapping[str, GroupStats]
    budget_used: Mapping[str, tuple[int, int | None]]


def _nearest_rank(ordered: Sequence[float], percent: int) -> float:
    """Return the nearest-rank ``percent`` percentile of ascending ``ordered`` (0 if empty)."""
    if not ordered:
        return 0.0
    rank = max(1, -(-percent * len(ordered) // 100))
    return ordered[rank - 1]


def _throughput_tokens(row: CallRecord) -> int:
    """Tokens a row produced: completions for generations, inputs for embeddings."""
    return row.completion_tokens if row.kind == "generate" else row.prompt_tokens


def _group_stats(rows: Sequence[CallRecord]) -> GroupStats:
    """Aggregate ``rows`` (non-empty) into one :class:`GroupStats`."""
    real = [r for r in rows if not r.cache_hit]
    seconds = sorted(r.seconds for r in real)
    total_seconds = sum(seconds)
    tokens = sum(_throughput_tokens(r) for r in real)
    reasons = Counter(reason for r in rows for _, reason in r.fallbacks)
    return GroupStats(
        calls=len(rows),
        cache_hit_pct=100.0 * (len(rows) - len(real)) / len(rows),
        p50_s=_nearest_rank(seconds, 50),
        p95_s=_nearest_rank(seconds, 95),
        max_s=seconds[-1] if seconds else 0.0,
        tokens_per_s=tokens / total_seconds if total_seconds > 0 else 0.0,
        failures_by_reason=MappingProxyType(dict(reasons)),
        fallbacks=sum(len(r.fallbacks) for r in rows),
        wait_s=float(sum(r.wait_seconds for r in rows)),
        cost=float(sum(r.cost or 0.0 for r in rows)),
    )


def _group_by(
    rows: Sequence[CallRecord], field: Callable[[CallRecord], str | None]
) -> Mapping[str, GroupStats]:
    """Group ``rows`` by ``field``; non-generation rows get a kind-qualified key."""
    groups: defaultdict[str, list[CallRecord]] = defaultdict(list)
    for row in rows:
        value = field(row)
        if row.kind != "generate":
            key = row.kind if value is None else f"{value}:{row.kind}"
        elif value is None:
            continue
        else:
            key = value
        groups[key].append(row)
    return MappingProxyType({key: _group_stats(group) for key, group in groups.items()})


def compute_summary(
    rows: Sequence[CallRecord], config: LLMConfig, budgets_used: Mapping[str, int]
) -> RunSummary:
    """Summarise ``rows`` per provider and per role, with each role's token budget.

    ``budgets_used`` maps a role to the tokens it used; the limit is the role's
    configured ``max_tokens_per_run`` (``None`` when unlimited).
    """
    budget = {
        role: (used, config.role_budget(role).max_tokens_per_run)
        for role, used in budgets_used.items()
    }
    return RunSummary(
        by_provider=_group_by(rows, lambda r: r.provider),
        by_role=_group_by(rows, lambda r: r.role),
        budget_used=MappingProxyType(budget),
    )
