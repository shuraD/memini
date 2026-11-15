"""Which queue entries may start now: dependencies merged, no path overlap, free slots.

Pure functions, no I/O: ``run-queue --workers N`` feeds them the queue and the state.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

from tools.orchestrator.state import paths_overlap


@dataclass(frozen=True)
class Candidate:
    """A queue entry that is neither merged nor running."""

    task_id: str
    paths: tuple[str, ...]
    depends_on: tuple[str, ...]


def pick(
    candidates: Sequence[Candidate],
    *,
    merged: Collection[str],
    running: Mapping[str, Sequence[str]],
    free: int,
) -> list[str]:
    """Return the task ids to start now, in queue order.

    An entry starts when every ``depends_on`` is merged and its paths overlap neither a running
    slice nor one picked earlier in this round, while slots are free.
    """
    taken = [tuple(p) for p in running.values()]
    out: list[str] = []
    for c in candidates:
        if len(out) >= free:
            break
        if any(d not in merged for d in c.depends_on):
            continue
        if any(paths_overlap(c.paths, other) for other in taken):
            continue
        out.append(c.task_id)
        taken.append(c.paths)
    return out
