"""Pre-merge gate: verify the review verdict for a commit (never trust it).

``reviews/<sha>.md`` must exist; its expected-vs-implemented table must have no
``missing`` row and every ``deferred`` row must carry a reason; no ``BLOCKING`` finding
may remain; a ``## Knowledge drift`` section must be present; the verdict must be
``MERGE: yes``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

from tools.ledger import _git_env

REVIEWS_DIR = Path(__file__).resolve().parents[2] / "reviews"
TABLE_COLUMNS = 3  # requirement | implemented where | status
_GIT = shutil.which("git") or "git"
# Markdown decoration a reviewer may put before a label: whitespace, quote/list/heading
# markers, numbering, task boxes, emphasis, code ticks, HTML tags, emoji and brackets.
_DECORATION = re.compile(r"^(?:\s|[>#*_`~+\-]|\d+[.)]|\[[ xX]?\]|<[^>]*>|[^\w\s])*")
_TAG = re.compile(r"[(\[][ \t]*BLOCKING[ \t]*[)\]]", re.IGNORECASE)
_FENCE = re.compile(r"^[ \t]*(```|~~~).*?^[ \t]*\1[ \t]*$", re.MULTILINE | re.DOTALL)
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_DRIFT = re.compile(r"^#{2,6}[ \t]+knowledge drift[ \t]*$", re.MULTILINE | re.IGNORECASE)
_VERDICT = re.compile(r"^MERGE:[ \t]*(\S+)[ \t]*$", re.MULTILINE | re.IGNORECASE)
_DEFERRED_WITH_REASON = re.compile(r"^deferred\s*\(\s*[^)\s]")


def _label(text: str) -> str:
    """Return ``text`` without leading markdown decoration, lower-cased."""
    return _DECORATION.sub("", text).strip().lower()


def _cells(line: str) -> list[str]:
    """Return the cells of a markdown table row (outer pipes optional), else ``[]``."""
    if "|" not in line:
        return []
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return cells if len(cells) >= TABLE_COLUMNS else []


def _marks_blocking(line: str) -> bool:
    """Return True when ``line`` marks a finding as BLOCKING.

    A line, or a table cell, marks it when it starts with the label after any
    decoration, or carries it as a tag ``(BLOCKING)`` / ``[BLOCKING]``; prose that
    only mentions the word ("no BLOCKING finding") does not.
    """
    if _TAG.search(line):
        return True
    return any(_label(part).startswith("blocking") for part in [line, *_cells(line)])


def problems(reviews_dir: Path, sha: str) -> list[str]:
    """Return every reason the review for ``sha`` does not allow a merge."""
    path = reviews_dir / f"{sha}.md"
    if not path.exists():
        return [f"no review file {path.name} in {reviews_dir}"]
    raw = path.read_text(encoding="utf-8")
    # structure (table, drift section, verdict) counts only outside code and comments;
    # a BLOCKING mark counts anywhere, hidden or not
    text = _FENCE.sub("", _COMMENT.sub("", raw))
    out: list[str] = []
    for line in text.splitlines():
        cells = _cells(line)
        if not cells or _label(cells[0]).startswith("spec requirement"):
            continue
        if all(set(c) <= set("-: ") for c in cells):  # header separator
            continue
        status = _label(cells[-1])
        if status.startswith("missing"):
            out.append(f"requirement {cells[0]!r} is missing")
        elif status.startswith("deferred") and not _DEFERRED_WITH_REASON.match(status):
            out.append(f"requirement {cells[0]!r} is deferred without a reason")
    if any(_marks_blocking(line) for line in raw.splitlines()):
        out.append("an unresolved BLOCKING finding remains")
    if not _DRIFT.search(text):
        out.append("no 'Knowledge drift' section (docs vs code not checked)")
    verdicts = {v.lower() for v in _VERDICT.findall(text)}
    if verdicts != {"yes"}:
        out.append("verdict is not 'MERGE: yes'")
    return out


def _git(repo: Path | None, *args: str) -> str:
    return subprocess.run(  # noqa: S603 — fixed args, no shell
        [_GIT, *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        env=_git_env(),
    ).stdout.strip()


_CHANGES = "ledgers/changes.sql"
_FILE_MODE = "100644"


def _is_review_record(repo: Path | None) -> bool:
    """Return True when HEAD is only the record of its parent's review.

    It may add or edit regular files under ``reviews/`` and append to
    ``ledgers/changes.sql`` (same mode, the earlier text an exact prefix); nothing else.
    """
    raw = _git(repo, "diff", "--raw", "--no-renames", "-z", "HEAD~1", "HEAD")
    fields = [f for f in raw.split("\0") if f]
    if not fields:
        return False
    for meta, path in zip(fields[::2], fields[1::2], strict=True):
        old_mode, new_mode, _, _, status = meta.lstrip(":").split()
        if path.startswith("reviews/"):
            ok = status in {"A", "M"} and new_mode == _FILE_MODE
        elif path == _CHANGES:
            ok = (
                status in {"A", "M"}
                and new_mode == _FILE_MODE
                and old_mode in {_FILE_MODE, "000000"}
                and _appended(repo, created=status == "A")
            )
        else:
            ok = False
        if not ok:
            return False
    return True


def _appended(repo: Path | None, *, created: bool) -> bool:
    """Return True when HEAD's ``changes.sql`` starts with HEAD~1's text unchanged."""
    before = "" if created else _git_raw(repo, "show", f"HEAD~1:{_CHANGES}")
    after = _git_raw(repo, "show", f"HEAD:{_CHANGES}")
    return after.startswith(before) and len(after) > len(before)


def _git_raw(repo: Path | None, *args: str) -> str:
    return subprocess.run(  # noqa: S603 — fixed args, no shell
        [_GIT, *args], cwd=repo, capture_output=True, text=True, check=True, env=_git_env()
    ).stdout


def reviewed_commit(repo: Path | None = None) -> str:
    """Return the short sha whose review the gate must verify.

    A review describes the tree of the commit it names; recording that review is
    itself a commit, so the head can never be the reviewed commit. When the head is
    such a record (see :func:`_is_review_record`) the reviewed commit is its parent;
    any other head is judged as itself (B-foundation-008).
    """
    head = _git(repo, "rev-parse", "--short=7", "HEAD")
    try:
        _git(repo, "rev-parse", "--verify", "HEAD~1")
    except subprocess.CalledProcessError:  # root commit
        return head
    if _is_review_record(repo):
        return _git(repo, "rev-parse", "--short=7", "HEAD~1")
    return head


def main(argv: list[str] | None = None) -> int:
    """Check the review of the reviewed commit (or ``--sha``); return 1 on any problem."""
    ap = argparse.ArgumentParser(prog="review_gate")
    ap.add_argument("--sha")
    ap.add_argument("--reviews", type=Path, default=REVIEWS_DIR)
    args = ap.parse_args(argv)
    sha = args.sha or reviewed_commit()
    found = problems(args.reviews, sha)
    for p in found:
        print(f"review gate failed: {p}")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
