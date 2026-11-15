# Sandbox image for running coder-written code (tests, gates, hold-out) off the host.
# Built from the repo's locked environment; retagged by the lock's hash when uv.lock changes
# (see tools/orchestrator/dockerexec.py). The slice's files are COPYed in per run, never
# mounted, and the container runs with --network none and an allow-listed environment.
# uv 0.13.6: patched for every uv advisory to date (0.9.9 had three); uv ships no bookworm image
# for it, so the base is Debian 13 (trixie), the current stable
FROM ghcr.io/astral-sh/uv:0.13.6-python3.12-trixie-slim

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    GIT_NO_REPLACE_OBJECTS=1

WORKDIR /build

# Only the dependency manifests, so the layer caches until they change. The environment is
# built at /opt/venv (outside /slice), because the slice is COPYed into a fresh tmpfs /slice
# per run, which would otherwise hide a venv built there. The project itself is a per-run
# export, so it is not installed here.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-editable

# a non-root user; /slice (the per-run tmpfs) and /tmp are the only writable paths
RUN useradd --create-home --uid 10001 runner && mkdir -p /slice && chown runner:runner /slice
USER runner
WORKDIR /slice
