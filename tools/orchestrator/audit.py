"""Parse the one-shot diff audit's answer; anything that is not a clear verdict is a reject.

The expected answer is a JSON object ``{"verdict": "pass" | "reject", "findings": [str]}``,
bare, inside a ```` ```json ```` fence, or embedded in prose. The first object that parses
and has a ``verdict`` decides. Unparseable or malformed output fails closed.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

_FENCE = re.compile(r"```(?:json|JSON)?[ \t]*\n(.*?)```", re.DOTALL)
UNPARSEABLE = "audit output is not a JSON verdict (fail closed)"


@dataclass(frozen=True)
class Verdict:
    """The audit's decision and its findings."""

    passed: bool
    findings: tuple[str, ...]


def _candidates(text: str) -> Iterator[Any]:
    """Yield every JSON value found: the whole text, each fence, each embedded object."""
    decoder = json.JSONDecoder()
    for chunk in (text, *_FENCE.findall(text)):
        with contextlib.suppress(ValueError):
            yield json.loads(chunk)
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                yield decoder.raw_decode(text, i)[0]
            except ValueError:
                continue


def parse(text: str) -> Verdict:
    """Return the verdict in ``text``; a missing or malformed verdict is a reject."""
    for value in _candidates(text):
        if not isinstance(value, dict) or "verdict" not in value:
            continue
        verdict, findings = value.get("verdict"), value.get("findings", [])
        if verdict not in ("pass", "reject"):
            return Verdict(False, (f"audit verdict {verdict!r} is neither pass nor reject",))
        if not isinstance(findings, list) or not all(isinstance(f, str) for f in findings):
            return Verdict(False, ("audit findings are not a list of strings",))
        if verdict == "reject":
            return Verdict(False, tuple(findings) or ("audit rejected without findings",))
        return Verdict(True, tuple(findings))
    return Verdict(False, (UNPARSEABLE,))
