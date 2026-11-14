"""Owner rule (2026-11-12): the checks are never changed to make something pass.

A commit that touches a protected check file must declare it with a non-empty
``CHECK-CHANGE: <reason>`` line in its message, so the change is visible in the log, the
ledger and the review, and carry the owner's approval
``OWNER-APPROVED-CHECK-CHANGE: <owner words, date>`` (owner 2026-11-12: no agent approves
its own gate change). The protected set itself lives in ``tools/ci/protected.txt``.
"""

from tools.ci import check_ledgers as cl

OWNER_OK = "OWNER-APPROVED-CHECK-CHANGE: owner 2026-11-12 'yes'\n"

PROTECTED_SAMPLE = [
    ".githooks/pre-commit",
    ".githooks/commit-msg",
    ".github/workflows/ci.yml",
    "tools/ci/check_ledgers.py",
    "tools/ci/protected.txt",
    "pyproject.toml",
    "docs/MERGE-CHECKLIST.md",
]


def test_every_check_file_is_protected() -> None:
    for path in PROTECTED_SAMPLE:
        assert cl.check_change_problems([path], "subject\n") != [], path


def test_touching_a_protected_file_without_declaring_it_is_rejected() -> None:
    problems = cl.check_change_problems(["tools/ci/check_ledgers.py", "src/x.py"], "subject\n")
    assert problems and "tools/ci/check_ledgers.py" in problems[0] and "CHECK-CHANGE" in problems[0]


def test_declared_check_change_passes_and_ordinary_files_need_nothing() -> None:
    msg = "subject\n\nCHECK-CHANGE: guard must fail loudly when git history is unreadable\n"
    assert cl.check_change_problems(["tools/ci/check_ledgers.py"], msg + OWNER_OK) == []
    assert cl.check_change_problems(["src/x.py", "tests/unit/test_x.py"], "subject\n") == []


def test_an_empty_check_change_reason_is_rejected() -> None:
    assert cl.check_change_problems([".githooks/pre-commit"], "s\n\nCHECK-CHANGE:   \n") != []


def test_a_directory_entry_protects_every_file_under_it() -> None:
    assert cl.check_change_problems([".githooks/pre-push"], "s\n") != []


def test_ci_checks_every_commit_of_the_branch_for_check_changes(tmp_path, monkeypatch) -> None:
    # review b0bacbd: CHECK-CHANGE was verified only for the last commit; a middle commit
    # touching a check file slipped through CI
    commits = [
        ("aaa1111", "first\n\nCHECK-CHANGE: ok\n" + OWNER_OK, [".githooks/pre-commit"]),
        ("bbb2222", "middle touches a check silently\n", ["tools/ci/check_ledgers.py"]),
        ("ccc3333", "last\n", ["src/x.py"]),
    ]
    problems = cl.check_change_problems_for_commits(commits)
    # every problem names the silent middle commit and its file; the declared ones pass
    assert problems
    assert all("bbb2222" in p and "check_ledgers.py" in p for p in problems)


def test_the_ledger_schema_itself_is_a_protected_check() -> None:
    # review b0bacbd: the ledger schema decides what the guards can see, so it is a check
    assert cl.check_change_problems(["ledgers/schema.sql"], "subject\n") != []


def test_a_check_change_without_the_owners_approval_is_rejected() -> None:
    # owner 2026-11-12: an agent changed the review gate to pass its own branch; a reason
    # line written by the agent is not enough
    msg = "subject\n\nCHECK-CHANGE: stricter only\n"
    problems = cl.check_change_problems(["tools/ci/review_gate.py"], msg)
    assert len(problems) == 1 and "OWNER-APPROVED-CHECK-CHANGE" in problems[0]


def test_owner_approval_alone_does_not_replace_the_reason() -> None:
    problems = cl.check_change_problems(["tools/ci/review_gate.py"], "subject\n\n" + OWNER_OK)
    assert len(problems) == 1 and "'CHECK-CHANGE: <reason>'" in problems[0]


def test_an_empty_owner_approval_is_rejected() -> None:
    msg = "s\n\nCHECK-CHANGE: reason\nOWNER-APPROVED-CHECK-CHANGE:   \n"
    assert cl.check_change_problems(["pyproject.toml"], msg) != []
