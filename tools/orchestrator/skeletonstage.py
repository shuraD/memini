"""The skeleton stage: the flow's steps 2 to 5 run by moves, with nobody carrying a message.

A scope is a version (the cards of its section of the Version file) or a feature (an explicit
card list given when the draft is asked for). Everything a project differs in is ``[skeleton]``
of the config: the roles, the documents and where they are written, the inputs, the size KPI.

- Step 2: a ``skeleton`` move to the author role. ``draft`` runs one agent WITH tools (it may
  write only the scope's documents) and then the machine check; a failing check gives one
  automatic revision, then the owner inbox. A person's comments arrive as the same move.
- Step 3: when the check passes ``draft`` posts a ``skeleton_review`` move to the reviewer role.
  ``review`` asks the model once: findings go back to the author as a ``skeleton`` move (bound
  by ``max_rounds``), ``ok`` writes the owner inbox entry.
- Step 4: ``approve`` records the owner's words with the content hash of the skeleton document.
- Step 5: ``stub`` queues the one ticket that creates the stubs, only for an approved content;
  ``stub_problems`` is that ticket's slice gate.

Every step writes an ops-ledger row.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from tools.orchestrator import (
    docfix,
    engine,
    enqueue,
    gitops,
    moves,
    planner,
    skeleton,
    state,
    vprnd,
)
from tools.orchestrator.config import Config
from tools.orchestrator.executor import executor_for

KIND = "skeleton"  # to the author: draft the documents, or revise them (the text is the comments)
REVIEW = "skeleton_review"  # to the reviewer: judge the skeleton against the cards
ENGINE_ROLE = "architect"  # the engine command of the author's run (config: engines.<name>)
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def task(scope: str) -> str:
    """Return the id the moves and ops rows of a scope's skeleton carry."""
    return f"{KIND}-{scope}"


def scope_cards(cfg: Config, scope: str) -> list[str]:
    """Return the explicit card list the scope was drafted with ([]: its Version section)."""
    asked = moves.for_task(cfg, task(scope), KIND)
    return list(asked[0]["payload"].get("cards", [])) if asked else []


def _versions(cfg: Config, scope: str, cards: Sequence[str]) -> dict[str, str]:
    """Return card -> version, where an explicit card list IS the scope."""
    found = docfix.versions(cfg)
    if not cards:
        return found
    return {**{c: v for c, v in found.items() if v != scope}, **dict.fromkeys(cards, scope)}


def source(repo: Path, rev: str) -> Callable[[str], str | None]:
    """Return the reader of a file's text at ``rev`` (None when the file is not there)."""

    def read(file: str) -> str | None:
        res = gitops.git(repo, "show", f"{rev}:{file}", check=False)
        return res.stdout if res.returncode == 0 else None

    return read


def digest(cfg: Config, scope: str) -> str:
    """Return the content hash of the scope's skeleton document."""
    return hashlib.sha256(cfg.skeleton.doc(KIND, scope).read_bytes()).hexdigest()


def check(
    cfg: Config, scope: str, cards: Sequence[str]
) -> tuple[list[skeleton.Row], skeleton.Report | None, list[str]]:
    """Run the machine check; return the rows, the report and every problem (none: it passes).

    A problem is a document that is not written, a card with no function, an inconsistent row,
    or existing code that differs from its row (a planned file is not expected in the code yet).
    """
    sk = cfg.skeleton
    absent = [p for p in (sk.doc(k, scope) for k in sk.documents) if not p.is_file()]
    problems = [f"{p.name} is not written" for p in absent]
    if sk.doc(KIND, scope) in absent:
        return [], None, problems
    rows, bad = skeleton.read(sk.doc(KIND, scope))
    squads = {c.id: c.squad for c in enqueue.cards(cfg).values()}
    try:
        report = skeleton.check(rows, scope, _versions(cfg, scope, cards), squads)
    except ValueError as exc:
        return rows, None, [*problems, str(exc)]
    report.problems[:0] = bad
    differs = skeleton.compare(rows, source(cfg.repo, cfg.main_branch))
    problems += [f"card {c} has no function" for c in report.missing]
    problems += report.problems
    problems += skeleton.unexplained(rows, differs, built=False)
    return rows, report, problems


def _size(rows: Sequence[skeleton.Row]) -> str:
    return f"{len(rows)} functions in {len({r.file for r in rows})} files"


def kpi(cfg: Config) -> str:
    """Return the size KPI text: the section of ``kpi_file`` under ``kpi_heading``."""
    sk = cfg.skeleton
    if sk.kpi_file is None or not sk.kpi_file.is_file():
        return "none given"
    text = sk.kpi_file.read_text(encoding="utf-8")
    if not sk.kpi_heading:
        return text.strip()
    parts = re.split(r"^#+ ", text, flags=re.MULTILINE)
    return next((p.strip() for p in parts if p.startswith(sk.kpi_heading)), "none given")


# ------------------------------------------------------------------ step 2: the author
def _run(cfg: Config, scope: str, cards: Sequence[str], feedback: str) -> list[str]:
    """Run the author's agent once; return the problems of the run itself.

    The agent's working directory is the repository (so it can run the read-only check); the
    guard hook lets it write only the scope's documents, read nothing private, and name only
    paths of the repository and the documents in a shell command. A change to the repository's
    files is a problem.
    """
    sk = cfg.skeleton
    name = engine.pick_engines(cfg, None)["qa"]
    if ENGINE_ROLE not in cfg.engines[name].roles:
        msg = f"engine {name} has no {ENGINE_ROLE} command"
        raise ValueError(msg)
    run_dir = cfg.state_dir / "runs" / task(scope)
    run_dir.mkdir(parents=True, exist_ok=True)
    n = len(list(run_dir.glob("*-settings.json"))) + 1
    docs = {k: sk.doc(k, scope) for k in sk.documents}
    sk.dir.mkdir(parents=True, exist_ok=True)
    settings_path = run_dir / f"{n:02d}-settings.json"
    settings = engine.settings(
        cfg,
        sk.dir,
        writable=[p.relative_to(sk.dir).as_posix() for p in docs.values()],
        guard_config=run_dir / f"{n:02d}-guard.json",
        protected=[],
        deny_read=sk.deny_read,
        bash_roots=[cfg.repo],
    )
    settings_path.write_text(engine.json_text(settings), encoding="utf-8")
    argv_check = ["uv", "run", "python", "-m", "tools.orchestrator", "skeleton-check", scope]
    argv_check += ["--doc", str(docs[KIND]), "--repo"]
    argv_check += ["--cards", ",".join(cards)] if cards else []
    prompt = engine.render(
        cfg,
        "architect_skeleton",
        scope=scope,
        cards=", ".join(cards) or f"every card line of the section {scope} of the Version file",
        mode="revise the documents" if docs[KIND].is_file() else "the first draft",
        version_file=str(cfg.version_file),
        cards_dir=str(cfg.cards_dir),
        inventories=str(sk.inventories),
        old_code=str(sk.old_code or "none"),
        repo=str(cfg.repo),
        kpi=kpi(cfg),
        documents="\n".join(f"- {p}" for p in docs.values()),
        feedback=feedback,
        check_command=" ".join(argv_check)
        + "  (before the stubs exist, a planned file or function reported as missing is expected)",
        **{f"doc_{k}": str(p) for k, p in docs.items()},
    )
    run_cfg = dataclasses.replace(cfg, max_budget_usd=sk.budget_usd or cfg.max_budget_usd)
    argv = engine.command(
        run_cfg, name, ENGINE_ROLE, workdir=cfg.repo, settings=settings_path, prompt=prompt
    )
    before = gitops.out(cfg.repo, "status", "--porcelain")
    res = executor_for("local", cfg.env_allow).run(
        argv,
        cwd=cfg.repo,
        stdin=prompt if cfg.engines[name].stdin else None,
        timeout=sk.timeout_s or cfg.engine_timeout_s,
        pass_env=cfg.engines[name].env,
    )
    text, cost = engine.agent_output(res.stdout)
    (run_dir / f"{n:02d}.log").write_text(
        f"exit {res.code}\ncost_usd {cost}\n{text}{res.stderr}", encoding="utf-8"
    )
    if gitops.out(cfg.repo, "status", "--porcelain") != before:
        return ["the run changed files of the repository: it may write only its documents"]
    return []


def draft(cfg: Config, move: moves.Move) -> str:
    """Handle a ``skeleton`` move: draft or revise, check, then the reviewer or the owner."""
    sk = cfg.skeleton
    scope = str(move["payload"]["scope"])
    cards = list(move["payload"].get("cards", []))
    who = move["task"]
    feedback = move["text"] if move["payload"].get("comments") else "none"
    try:
        if sk.doc(KIND, scope).is_file():  # a revision starts from the current check output
            now = check(cfg, scope, cards)[2]
            feedback += "\nThe machine check now:\n- " + ("\n- ".join(now) or "passes")
        for attempt in (1, 2):  # one automatic revision, like the spec stage's one retry
            ran = _run(cfg, scope, cards, feedback)
            rows, report, problems = check(cfg, scope, cards)
            problems = [*ran, *problems]
            state.ops_log(
                cfg,
                who,
                "skeleton_drafted",
                f"round {attempt} for {move['from']}: {_size(rows)}; "
                + ("check ok" if not problems else f"check failed ({len(problems)})"),
                move=move["id"],
                functions=len(rows),
                files=len({r.file for r in rows}),
                missing=len(report.missing) if report else None,
                extra=len(report.extra) if report else None,
                problems=problems[:20],
            )
            if not problems:
                break
            feedback = "Fix only these problems:\n- " + "\n- ".join(problems)
    except Exception as exc:  # an open move would repeat the agent run every cycle
        problems = [f"the run failed ({type(exc).__name__}: {exc})"]
    if problems:
        found = "; ".join(problems)[:1500]
        vprnd._escalate(cfg, {**move, "text": f"the skeleton of {scope} fails its check: {found}"})
        return f"check failed: {found}"
    moves.post(
        cfg,
        who,
        sender=sk.author,
        to=sk.reviewer,
        kind=REVIEW,
        text=f"judge the skeleton of {scope} against its cards ({_size(rows)}, check ok)",
        scope=scope,
        cards=cards,
    )
    return f"drafted: {_size(rows)}; check ok; sent to {sk.reviewer}"


# ------------------------------------------------------------------ step 3: the reviewer
def _card_lines(cfg: Config, scope: str, cards: Sequence[str]) -> str:
    """Return the Version file's lines of the scope's cards."""
    wanted = _versions(cfg, scope, cards)
    lines = cfg.version_file.read_text(encoding="utf-8").splitlines()
    return "\n".join(x for x in lines if (m := docfix._ROW.match(x)) and wanted.get(m[1]) == scope)


def review(cfg: Config, move: moves.Move) -> str:
    """Handle a ``skeleton_review`` move: one model call; ``ok`` or findings back to the author."""
    sk = cfg.skeleton
    scope = str(move["payload"]["scope"])
    cards = list(move["payload"].get("cards", []))
    who = move["task"]
    try:
        rows, report, problems = check(cfg, scope, cards)
        if problems or report is None:
            findings = [f"the machine check fails: {p}" for p in problems]
        else:
            prompt = engine.render(
                cfg,
                REVIEW,
                scope=scope,
                cards=_card_lines(cfg, scope, cards),
                check=skeleton.render(report),
                rows="\n".join(
                    f"{r.id} | {r.file} | {r.name} | {r.signature} | "
                    f"{', '.join(r.cards) or '-'} | {r.note}"
                    for r in rows
                ),
            )
            verdict = json.loads(re.findall(r"\{.*\}", planner._ask(cfg, prompt), re.DOTALL)[0])
            findings = [vprnd.one_line(f) for f in verdict.get("findings") or []]
            if verdict["verdict"] != "ok" and not any(findings):
                findings = ["the reviewer gave no usable finding"]
            findings = [] if verdict["verdict"] == "ok" else findings
    except Exception as exc:  # an open move would repeat the model call every cycle
        text = f"the review of the skeleton of {scope} failed ({type(exc).__name__})"
        vprnd._escalate(cfg, {**move, "text": text})
        return text
    if not findings:
        sha = digest(cfg, scope)
        state.ops_log(cfg, who, "skeleton_reviewed", f"ok: {_size(rows)}", sha256=sha)
        paths = ", ".join(str(sk.doc(k, scope)) for k in sk.documents)
        state.inbox_append(
            cfg,
            who,
            f"skeleton {scope} is ready for approval",
            2,
            f"{_size(rows)}; machine check ok; {sk.reviewer}: ok. Size KPI: {kpi(cfg)} "
            f"Documents: {paths}. Content hash {sha}. To approve: skeleton-approve {scope} "
            "--words '<your dated words>' --ticket <id> --file <path>.",
        )
        return "ok"
    text = "; ".join(findings)[:1500]
    asked = moves.for_task(cfg, who, KIND)
    last_person = max((i for i, m in enumerate(asked) if m["from"] != sk.reviewer), default=-1)
    automatic = len(asked) - 1 - last_person  # the reviewer's rounds since a person last spoke
    state.ops_log(cfg, who, "skeleton_findings", text, round=automatic + 1)
    if automatic >= cfg.max_rounds - 1:
        vprnd._escalate(
            cfg, {**move, "text": f"the skeleton of {scope} is at its round limit: {text}"}
        )
        return f"escalated to the owner at the round limit: {text}"
    moves.post(
        cfg,
        who,
        sender=sk.reviewer,
        to=sk.author,
        kind=KIND,
        text=text,
        scope=scope,
        cards=cards,
        comments=True,
    )
    return f"findings: {text}"


# ------------------------------------------------------------------ steps 4 and 5
def approved(cfg: Config, scope: str) -> bool:
    """Return True when the owner approved the skeleton document's current content."""
    doc = cfg.skeleton.doc(KIND, scope)
    rows = state.ops_rows(cfg, task(scope), "skeleton_approved")
    return doc.is_file() and any(r["details"].get("sha256") == digest(cfg, scope) for r in rows)


def approve(cfg: Config, scope: str, words: str) -> str:
    """Record the owner's approval (the only way one is recorded); return the content hash."""
    if not _DATE.search(words):
        msg = "words must quote the owner and give the date (YYYY-MM-DD)"
        raise ValueError(msg)
    problems = check(cfg, scope, scope_cards(cfg, scope))[2]
    if problems:
        msg = f"the skeleton of {scope} fails its check: " + "; ".join(problems)[:1000]
        raise ValueError(msg)
    sha = digest(cfg, scope)
    seen = any(
        r["details"].get("sha256") == sha
        for r in state.ops_rows(cfg, task(scope), "skeleton_reviewed")
    )
    state.ops_log(cfg, task(scope), "skeleton_approved", words, sha256=sha, reviewed=seen)
    return sha


def stub(cfg: Config, scope: str, task_id: str, module: str) -> list[str]:
    """Queue the one ticket that creates the stubs; return why it was refused (none: queued)."""
    doc = cfg.skeleton.doc(KIND, scope)
    if not approved(cfg, scope):
        return [f"no skeleton_approved row for the current content of {doc}"]
    rows, _ = skeleton.read(doc)
    text = engine.render(
        cfg,
        "skeleton_stub",
        scope=scope,
        functions=str(len(rows)),
        files=str(len({r.file for r in rows})),
    )
    _, found = enqueue.enqueue(
        cfg, task_id=task_id, module=module, text=" ".join(text.split()), sources=[str(doc)]
    )
    return found


def stub_problems(cfg: Config, scope: str, read: Callable[[str], str | None]) -> list[str]:
    """Return why a slice's tree (``read``) is not the approved skeleton of ``scope`` (the gate)."""
    if not approved(cfg, scope):
        return [f"skeleton gate: the document of {scope} is not approved in its current content"]
    rows, _ = skeleton.read(cfg.skeleton.doc(KIND, scope))
    differs = skeleton.unexplained(rows, skeleton.compare(rows, read), built=True)
    return [f"skeleton gate: {d}" for d in differs]
