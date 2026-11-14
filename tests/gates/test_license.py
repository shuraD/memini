"""Gate: the project ships the unmodified AGPL-3.0 and declares it (T-061).

Owner decision 2026-11-13: the LICENSE file is the unmodified AGPL-3.0 text, and pyproject
declares ``AGPL-3.0-or-later``. This gate fails without the LICENSE file, with a LICENSE that
is not the AGPL-3.0 text, or with a different pyproject license.

"Unmodified" is checked offline by the AGPL-3.0's own verbatim, canonical strings (its title,
version date, FSF copyright, the "changing it is not allowed" verbatim-copy clause, the
Affero-only section 13 "Remote Network Interaction", and the warranty/liability sections) plus
a full-text length floor — a GPL/MIT text or a stub fails these. A byte-exact check against the
FSF master would need that text fetched at release time, which is an owner/DevOps step.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
LICENSE = ROOT / "LICENSE"
PYPROJECT = ROOT / "pyproject.toml"

# Verbatim strings from the AGPL-3.0 (Free Software Foundation, 19 November 2007).
AGPL_MARKERS = [
    "GNU AFFERO GENERAL PUBLIC LICENSE",
    "Version 3, 19 November 2007",
    "Copyright (C) 2007 Free Software Foundation, Inc.",
    "Everyone is permitted to copy and distribute verbatim copies",
    "but changing it is not allowed.",
    "Remote Network Interaction",  # AGPL section 13 — the clause that distinguishes it from GPL
    "Disclaimer of Warranty",
    "Limitation of Liability",
]
# The full AGPL-3.0 plain text is ~34 KB; a stub, a link, or a shorter licence fails this.
_MIN_LICENSE_LEN = 30000
EXPECTED_SPDX = "AGPL-3.0-or-later"


def _license_text() -> str:
    assert LICENSE.is_file(), "LICENSE file is missing; it must be the unmodified AGPL-3.0 text"
    return LICENSE.read_text(encoding="utf-8")


def _pyproject_license() -> Any:
    data: dict[str, Any] = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    field = data.get("project", {}).get("license")
    return field.get("text") if isinstance(field, dict) else field


def test_license_file_exists() -> None:
    """A LICENSE file must be present at the repo root."""
    assert LICENSE.is_file(), "no LICENSE file at the repo root"


@pytest.mark.parametrize("marker", AGPL_MARKERS)
def test_license_is_the_unmodified_agpl_3_text(marker: str) -> None:
    """Every canonical AGPL-3.0 string is present, so the text is AGPL-3.0 and not altered."""
    assert marker in _license_text(), f"LICENSE is missing the AGPL-3.0 marker: {marker!r}"


def test_license_is_the_full_text_not_a_stub() -> None:
    """The LICENSE is the full AGPL-3.0 text, not a stub or a link."""
    assert len(_license_text()) >= _MIN_LICENSE_LEN, (
        "LICENSE is too short to be the full AGPL-3.0 text"
    )


def test_pyproject_declares_agpl_3_or_later() -> None:
    """pyproject's project.license is exactly AGPL-3.0-or-later."""
    value = _pyproject_license()
    assert value == EXPECTED_SPDX, f"pyproject license is {value!r}, expected {EXPECTED_SPDX!r}"
