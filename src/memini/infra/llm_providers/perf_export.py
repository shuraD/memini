"""Export the call log to a JSON Lines file, one object per row.

Each line is a JSON object whose keys are exactly the :class:`CallRecord` field
names, so a record can be rebuilt from it. The log holds no prompt or reply text,
so none can reach the export.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

from memini.infra.llm_providers.calllog import CallLog

__all__ = ["export_jsonl"]


def export_jsonl(log: CallLog, path: str | os.PathLike[str]) -> int:
    """Write every row of ``log`` to ``path`` as JSON Lines, ordered by ``ts``.

    An existing file is replaced, never appended to; an empty log gives an empty
    file. Rows with equal ``ts`` keep their insertion order.

    Returns:
        The number of rows written.

    """
    records = sorted(log.rows(), key=lambda rec: rec.ts)
    with Path(path).open("w", encoding="utf-8", newline="\n") as out:
        for rec in records:
            out.write(json.dumps(dataclasses.asdict(rec), ensure_ascii=False))
            out.write("\n")
    return len(records)
