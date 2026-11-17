"""Request channel between squads: one inbox file per squad in ``<state_dir>/agent-inbox/``.

A coder (dev1..dev4) posts a request to ``qa`` or ``devops``; the addressee answers in the same
file. Nobody edits the files by hand: use the commands (run from any checkout)::

    python -m tools.orchestrator.agent_inbox post  --from dev1 --to qa "what the request is"
    python -m tools.orchestrator.agent_inbox reply --from qa --id M-0007 "the answer"
    python -m tools.orchestrator.agent_inbox unread --who dev1 [--mark]

``--from`` and ``--who`` default to ``$MEMINI_SQUAD``. The hook runs ``unread --mark`` at session
start and on each prompt, so a session sees each message once. Every post, answer and read is
logged with its time in the ops ledger. Anything that needs the owner (T2) is not posted here: it
goes through vp-rnd, and this tool refuses any other addressee.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from tools.orchestrator import config, state

CODERS = ("dev1", "dev2", "dev3", "dev4")
TARGETS = ("qa", "devops")
_HEAD = re.compile(r"^## (M-\d{4}) (\S+) (\S+) -> (\S+?)(?: re (M-\d{4}))?$", re.MULTILINE)


@dataclass(frozen=True)
class Message:
    """One entry of an inbox file."""

    id: str
    at: str
    sender: str
    to: str
    re: str
    text: str
    file: str


def _dir(cfg: config.Config) -> Path:
    return cfg.state_dir / "agent-inbox"


def _messages(cfg: config.Config) -> list[Message]:
    found: list[Message] = []
    for path in sorted(_dir(cfg).glob("*.md")):
        body = path.read_text(encoding="utf-8")
        heads = list(_HEAD.finditer(body))
        for i, head in enumerate(heads):
            end = heads[i + 1].start() if i + 1 < len(heads) else len(body)
            found.append(
                Message(
                    head[1],
                    head[2],
                    head[3],
                    head[4],
                    head[5] or "",
                    body[head.end() : end].strip(),
                    path.stem,
                )
            )
    return sorted(found, key=lambda m: m.id)


def _append(
    cfg: config.Config, file: str, sender: str, to: str, text: str, *, re_id: str = ""
) -> str:
    """Write one entry to ``<file>.md`` and log it; return its id."""
    text = text.strip()
    if not text:
        raise ValueError("the message is empty")
    text = re.sub(r"^(## M-)", r" \1", text, flags=re.MULTILINE)  # a body cannot fake a header
    with state.locked(cfg.state_dir / "agent-inbox.lock"):
        _dir(cfg).mkdir(parents=True, exist_ok=True)
        last = max((int(m.id[2:]) for m in _messages(cfg)), default=0)
        msg_id = f"M-{last + 1:04d}"
        at = datetime.now(UTC).isoformat(timespec="seconds")
        head = f"## {msg_id} {at} {sender} -> {to}" + (f" re {re_id}" if re_id else "")
        with (_dir(cfg) / f"{file}.md").open("a", encoding="utf-8") as fh:
            fh.write(f"{head}\n{text}\n\n")
    state.ops_log(cfg, msg_id, "inbox_reply" if re_id else "inbox_post", text, sender=sender, to=to)
    return msg_id


def post(cfg: config.Config, sender: str, to: str, text: str) -> str:
    """Post a coder's request to QA or DevOps, into the addressee's file."""
    if sender not in CODERS:
        raise ValueError(f"only {', '.join(CODERS)} post requests, not {sender!r}")
    if to not in TARGETS:
        raise ValueError(f"post to {' or '.join(TARGETS)}; T2 items go to the owner through vp-rnd")
    return _append(cfg, to, sender, to, text)


def notify(cfg: config.Config, to: str, text: str, sender: str = "orchestrator") -> str:
    """Post a message from the orchestrator itself (the ready gate, a stopped slice)."""
    return _append(cfg, to, sender, to, text)


def reply(cfg: config.Config, sender: str, msg_id: str, text: str) -> str:
    """Answer a request in the same file, addressed to its sender."""
    request = next((m for m in _messages(cfg) if m.id == msg_id and not m.re), None)
    if request is None:
        raise ValueError(f"no request {msg_id}")
    if sender != request.to:
        raise ValueError(f"{msg_id} is addressed to {request.to}; only {request.to} answers it")
    return _append(cfg, request.file, sender, request.sender, text, re_id=msg_id)


def unread(cfg: config.Config, who: str, *, mark: bool = False) -> list[Message]:
    """Messages addressed to ``who`` that its sessions have not been shown yet."""
    cursor = _dir(cfg) / f".read-{who}"
    seen = cursor.read_text(encoding="utf-8").strip() if cursor.exists() else "M-0000"
    new = [m for m in _messages(cfg) if m.to == who and m.id > seen]
    if mark and new:
        cursor.write_text(new[-1].id, encoding="utf-8")
        state.ops_log(cfg, new[-1].id, "inbox_read", f"{who} read {len(new)}", who=who)
    return new


def render(who: str, messages: Sequence[Message]) -> str:
    """Return the text a session sees: each message, then how to answer."""
    if not messages:
        return ""
    lines = [f"[agent-inbox] {len(messages)} unread for {who}:"]
    for m in messages:
        lines.append(f"{m.id} {m.at} {m.sender} -> {m.to}" + (f" re {m.re}" if m.re else ""))
        lines.append(m.text)
    cmd = "python -m tools.orchestrator.agent_inbox"
    if who in CODERS:
        lines.append(f'Ask QA or DevOps: {cmd} post --from {who} --to qa "..."')
    else:
        lines.append(f'Answer: {cmd} reply --from {who} --id <M-id> "..."')
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: post, reply, unread."""
    ap = argparse.ArgumentParser(prog="agent_inbox", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    who = os.environ.get("MEMINI_SQUAD", "")
    for name in ("post", "reply"):
        p = sub.add_parser(name)
        p.add_argument("--from", dest="sender", default=who)
        if name == "post":
            p.add_argument("--to", required=True)
        else:
            p.add_argument("--id", required=True)
        p.add_argument("text")
    p = sub.add_parser("unread")
    p.add_argument("--who", default=who)
    p.add_argument("--mark", action="store_true")
    args = ap.parse_args(argv)
    cfg = config.load_config(Path(__file__).resolve().parents[2])
    try:
        if args.cmd == "post":
            print(post(cfg, args.sender, args.to, args.text))
        elif args.cmd == "reply":
            print(reply(cfg, args.sender, args.id, args.text))
        elif args.who:  # no squad set: a hook prints nothing
            sys.stdout.write(render(args.who, unread(cfg, args.who, mark=args.mark)))
    except ValueError as exc:
        print(f"agent_inbox: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
