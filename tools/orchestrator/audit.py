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


@dataclass(frozen=True)
class Verification:
    """The Verifier's table, and what blocks the merge (D8): code problems go to the coder."""

    table: str
    code_problems: tuple[str, ...]
    test_problems: tuple[str, ...]
    extras: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Return True when every card line is met, reachable and proved by a test."""
        return not self.code_problems and not self.test_problems


_HEAD = (
    "| Card ID | status | code (file:line) | reachable | test that proves it"
    " | not done as the old code did |\n|---|---|---|---|---|---|\n"
)


def parse_verifier(text: str, cards: list[str]) -> Verification:
    """Return the Verifier's table for ``cards``; anything unclear fails closed.

    The answer is ``{"lines": [{"card", "status": met|partial|missing, "code", "reachable",
    "test", "gap"}], "extras": [str]}``. A card without a row, a status other than ``met``, an
    unreachable line is a code problem; a ``met`` line without a test is a test problem.
    """
    answer = next(
        (v for v in _candidates(text) if isinstance(v, dict) and isinstance(v.get("lines"), list)),
        None,
    )
    if answer is None:
        return Verification("", ("verifier output is not a JSON table (fail closed)",), (), ())
    rows = {str(r.get("card")): r for r in answer["lines"] if isinstance(r, dict)}
    table, code, tests = _HEAD, [], []
    for card in cards:
        row = rows.get(card)
        if row is None:
            code.append(f"{card}: the verifier reported no row (fail closed)")
            table += f"| {card} | missing | - | - | - | not reported |\n"
            continue
        status, where = str(row.get("status")), str(row.get("code") or "-")
        reach, test, gap = (
            bool(row.get("reachable")),
            str(row.get("test") or ""),
            str(row.get("gap") or ""),
        )
        seen = "yes" if reach else "no"
        table += f"| {card} | {status} | {where} | {seen} | {test or '-'} | {gap or '-'} |\n"
        if status != "met":
            code.append(f"{card} is {status}: {gap}"[:300])
        elif not reach:
            code.append(f"{card} is not reachable from the entry point ({where})")
        elif not test:
            tests.append(f"{card} is met in {where} but no test proves it")
    extras = tuple(str(x) for x in answer.get("extras") or () if isinstance(x, str))
    return Verification(table, tuple(code), tuple(tests), extras)
