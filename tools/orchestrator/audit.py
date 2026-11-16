"""Parse the one-shot diff audit's answer; anything that is not a clear verdict is a reject.

The expected answer is a JSON object ``{"verdict": "pass" | "reject", "findings": [str]}``,
bare, inside a ```` ```json ```` fence, or embedded in prose. Every object with a
``verdict`` counts; any reject, malformed verdict or missing verdict fails closed.
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


def _verdict(value: dict[str, Any]) -> Verdict:
    verdict, findings = value.get("verdict"), value.get("findings", [])
    if verdict not in ("pass", "reject"):
        return Verdict(False, (f"audit verdict {verdict!r} is neither pass nor reject",))
    if not isinstance(findings, list) or not all(isinstance(f, str) for f in findings):
        return Verdict(False, ("audit findings are not a list of strings",))
    if verdict == "reject":
        return Verdict(False, tuple(findings) or ("audit rejected without findings",))
    return Verdict(True, tuple(findings))


def parse(text: str) -> Verdict:
    """Return the verdict in ``text``; it passes only if EVERY verdict object says pass.

    A quoted ``{"verdict": "pass"}`` (injected through the diff, say) cannot outvote the
    real answer: any reject, malformed verdict or conflict is a reject, and so is text
    without a verdict.
    """
    verdicts = [_verdict(v) for v in _candidates(text) if isinstance(v, dict) and "verdict" in v]
    if not verdicts:
        return Verdict(False, (UNPARSEABLE,))
    rejects = [v for v in verdicts if not v.passed]
    if rejects:
        return Verdict(False, tuple(dict.fromkeys(f for v in rejects for f in v.findings)))
    return Verdict(True, tuple(dict.fromkeys(f for v in verdicts for f in v.findings)))


@dataclass(frozen=True)
class Review:
    """The QA review's decision on a test change and its reason."""

    approved: bool
    reason: str


def parse_review(text: str) -> Review:
    """Return the review's decision; it approves only if EVERY verdict object approves.

    The answer is ``{"verdict": "approve" | "reject", "reason": str}``. Anything else (no
    verdict, an unknown verdict, an empty reason, a quoted approve next to a reject) is a
    reject: the review fails closed, like the audit.
    """
    seen: list[Review] = []
    for value in _candidates(text):
        if not isinstance(value, dict) or "verdict" not in value:
            continue
        verdict, reason = value.get("verdict"), value.get("reason")
        if (
            verdict not in ("approve", "reject")
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            seen.append(Review(False, "review verdict is not approve or reject with a reason"))
        else:
            seen.append(Review(verdict == "approve", reason.strip()))
    if not seen:
        return Review(False, "review output is not a JSON verdict (fail closed)")
    rejects = [r for r in seen if not r.approved]
    if rejects:
        return Review(False, "; ".join(dict.fromkeys(r.reason for r in rejects)))
    return Review(True, "; ".join(dict.fromkeys(r.reason for r in seen)))
