"""Well-known file locations, resolved in one place.

Configuration is resolved in this order:

1. ``MEMINI_CONFIG_DIR``, when set (an explicit override for any install).
2. ``memini/config/`` inside the package: an installed wheel carries ``config/`` there
   (``pyproject.toml`` force-include, T-082).
3. ``config/`` at the repository root: a source checkout, where the package is ``src/memini``
   and the configuration is not copied into it.

Nothing in ``memini.core`` computes a filesystem path itself (architecture: configuration
belongs to ``infra``).
"""

from __future__ import annotations

import os
from pathlib import Path

_PACKAGE_CONFIG = Path(__file__).resolve().parents[1] / "config"
_REPO_CONFIG = Path(__file__).resolve().parents[3] / "config"


def config_dir() -> Path:
    """Return the configuration directory: the override, the packaged copy, or the repo's."""
    override = os.environ.get("MEMINI_CONFIG_DIR")
    if override:
        return Path(override)
    return _PACKAGE_CONFIG if _PACKAGE_CONFIG.is_dir() else _REPO_CONFIG


def schema_file() -> Path:
    """Return the graph schema file."""
    return config_dir() / "graph_schema.yaml"


def prompts_dir() -> Path:
    """Return the directory holding every prompt text file."""
    return config_dir() / "prompts"
