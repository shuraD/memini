"""Private cache stores that never share a file with any other store.

:func:`private_store` opens a new file :class:`Store` in a fresh, uniquely named
subdirectory of a given directory, so a run (a benchmark, say) can cache and clear
its own rows without touching a shared store.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from memini.infra.llm_providers.store import Store

__all__ = ["private_store"]

_FILE_NAME = "cache.sqlite"


def private_store(directory: str | os.PathLike[str]) -> Store:
    """Open a new, empty file store in a fresh unique subdirectory of ``directory``.

    Each call creates a different subdirectory (mode 0700), so the file never coincides
    with another store's; the store keeps the file at mode 0600.

    Raises:
        FileNotFoundError: when ``directory`` does not exist.

    """
    parent = Path(directory)
    if not parent.is_dir():
        raise FileNotFoundError(f"directory does not exist: {parent}")
    subdirectory = parent / uuid.uuid4().hex
    subdirectory.mkdir(mode=0o700, exist_ok=False)
    return Store(subdirectory / _FILE_NAME)
