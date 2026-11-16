"""The ``docker`` executor: run coder-written code in a fresh, network-less container.

Every command the pipeline sends through the executor (the coder's own pytest via the
sandbox wrapper, the gates, the QA tests and the hold-out) runs in its own
``docker run --rm`` container built from the repo's locked image. The container gets:

- ``--network none`` (no network at all), or, when ``sandbox_network`` is configured, only
  the internal Docker network that holds the test Neo4j (no internet);
- no host environment: only the executor's allow-list, passed explicitly;
- no mount of ``$HOME`` or of the repo's ``.git``; the working directory is COPYed into the
  image at build-of-context time, not bind-mounted, so the container cannot reach the host
  tree, and a read-only root filesystem with a writable ``/tmp`` and ``/slice``;
- ``--user`` non-root, dropped capabilities, no new privileges, a pids and memory limit.

The image is built from ``tools/orchestrator/Dockerfile`` and tagged by the hash of
``uv.lock`` (rebuilt only when the lock changes). Results come back as the process exit code
and whatever the command wrote under the working directory (e.g. a junit report), which the
caller reads on the host after the container is gone.
"""

from __future__ import annotations

import hashlib
import io
import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from tools.orchestrator.config import Config
from tools.orchestrator.executor import NO_REPLACE, Result, allowed_env

_CONTEXT = ("pyproject.toml", "uv.lock")
WORKDIR = "/slice"
_EXTRAS = ".sandbox"  # host files a command names (outside the slice), shipped in the tar
# host-only cache locations: meaningless (and read-only) inside the container
_HOST_ONLY_ENV = ("XDG_CACHE_HOME", "UV_PYTHON_INSTALL_DIR")
# the container's own writable places (/tmp is its tmpfs); uv uses the image venv as it is
_CONTAINER_ENV = {
    "HOME": "/tmp",  # noqa: S108  # nosec B108 — the container's tmpfs, not a host directory
    "TMPDIR": "/tmp",  # noqa: S108  # nosec B108
    "UV_CACHE_DIR": "/tmp/uv-cache",  # noqa: S108  # nosec B108
    "UV_NO_SYNC": "1",
    "UV_OFFLINE": "1",
}


class DockerError(RuntimeError):
    """The docker CLI could not build the image or run a container."""


def lock_tag(dockerfile: Path, lock: Path) -> str:
    """Return the image tag: a hash of the Dockerfile and the lock (rebuild when either moves)."""
    digest = hashlib.sha256()
    for path in (dockerfile, lock):
        digest.update(path.read_bytes() if path.is_file() else b"")
    return digest.hexdigest()[:16]


def docker_run_argv(
    cli: str,
    image: str,
    *,
    workdir: str,
    env: Mapping[str, str],
    memory: str,
    pids: int,
    network: str = "none",
) -> list[str]:
    """Return the ``docker run`` argv with the sandbox hardening flags (no command yet).

    The flags are asserted by the unit tests: no network, a read-only root with writable
    ``/tmp`` and workdir, a non-root user, dropped capabilities, no new privileges, limits,
    and only the environment variables in ``env`` (never the host's). ``network`` is ``none``
    unless the caller names the internal test-Neo4j network.
    """
    argv = [
        cli,
        "run",
        "--rm",
        "-i",  # the slice tar arrives on stdin
        "--network",
        network,
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev",  # noqa: S108  # nosec B108 — container tmpfs path, not a host dir
        "--tmpfs",
        f"{workdir}:rw,nosuid,nodev,uid=10001,gid=10001",
        "--workdir",
        workdir,
        "--user",
        "10001:10001",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        str(pids),
        "--memory",
        memory,
        "--env-file",
        "/dev/null",  # start from an empty environment
    ]
    for key in sorted(env):
        argv += ["--env", f"{key}={env[key]}"]
    argv.append(image)
    return argv


class DockerExecutor:
    """Run each command in a fresh sandbox container built on a copy of the working directory."""

    def __init__(
        self,
        *,
        cli: str,
        image: str,
        dockerfile: Path,
        lock: Path,
        env_allow: Sequence[str],
        memory: str = "2g",
        pids: int = 512,
        build_timeout: int = 1800,
        network: str = "",
        neo4j_uri: str = "",
        neo4j_user: str = "",
    ) -> None:
        """Configure the executor; the image is built lazily on the first run."""
        self.cli = cli
        self.dockerfile = dockerfile
        self.lock = lock
        self.env_allow = tuple(env_allow)
        self.memory = memory
        self.pids = pids
        self.build_timeout = build_timeout
        self.network = network
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.tag = f"{image}:{lock_tag(dockerfile, lock)}"
        self._built = False

    # -------------------------------------------------------------- image
    def ensure_image(self) -> None:
        """Build the sandbox image for the current lock hash unless it already exists."""
        if self._built or self._image_exists():
            self._built = True
            return
        with tempfile.TemporaryDirectory(prefix="memini-dockerctx-") as tmp:
            ctx = Path(tmp)
            shutil.copyfile(self.dockerfile, ctx / "Dockerfile")
            for name in _CONTEXT:
                shutil.copyfile(self.lock.parent / name, ctx / name)
            res = self._docker(
                ["build", "-t", self.tag, "-f", str(ctx / "Dockerfile"), str(ctx)],
                timeout=self.build_timeout,
            )
        if res.returncode != 0:
            msg = f"sandbox image build failed: {(res.stderr or res.stdout).strip()[-800:]}"
            raise DockerError(msg)
        self._built = True

    def _image_exists(self) -> bool:
        return self._docker(["image", "inspect", self.tag], timeout=60).returncode == 0

    def _docker(self, args: Sequence[str], timeout: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 — fixed docker CLI, our own args, no shell
            [self.cli, *args],
            capture_output=True,
            text=True,
            env={**_host_path(), **NO_REPLACE},
            timeout=timeout,
            check=False,
        )

    # -------------------------------------------------------------- run
    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        stdin: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout: int = 1800,
        pass_env: Sequence[str] = (),
    ) -> Result:
        """Run ``argv`` in a fresh container whose ``/slice`` is a copy of ``cwd``.

        The command's combined output and its exit code come back; nothing is mounted. The
        slice is piped in as a tar on stdin and unpacked into the container's own ``/slice``
        tmpfs, so the host tree and ``.git`` are never reachable. A result the command needs
        in-process (the junit report) is written and read inside the container by
        ``tools/ci/pytest_gate.py``; the host reads only ``output`` and ``code``.
        """
        try:
            self.ensure_image()
        except (DockerError, OSError, subprocess.SubprocessError) as exc:
            return Result(125, "", f"docker image unavailable: {exc}\n")
        # the container provides its own PATH; every other host path is replaced by a container
        # one (a host path inside the read-only container is at best missing, at worst a leak)
        forward = [k for k in (*self.env_allow, *pass_env) if k not in ("PATH", "HOME")]
        container_env = _container_env(allowed_env(forward, env), cwd)
        if self.network:
            password = os.environ.get("NEO4J_PASSWORD", "")
            if not password:
                return Result(125, "", "sandbox network is set but NEO4J_PASSWORD is not\n")
            # fixed by the orchestrator: a coder-supplied value never names another host
            container_env.update(
                NEO4J_URI=self.neo4j_uri, NEO4J_USER=self.neo4j_user, NEO4J_PASSWORD=password
            )
        extras: list[tuple[str, Path]] = []
        run_argv = docker_run_argv(
            self.cli,
            self.tag,
            workdir=WORKDIR,
            env=container_env,
            memory=self.memory,
            pids=self.pids,
            network=self.network or "none",
        )
        prelude = f"tar xf - -C {WORKDIR} && "  # unpack the slice piped on stdin
        command = " ".join(_sh_quote(a) for a in _container_argv(argv, cwd, extras))
        full = [*run_argv, "bash", "-c", f"{prelude}{command}"]
        try:
            payload = _slice_tar(cwd, extras)
            res = subprocess.run(  # noqa: S603 — docker CLI + our argv, no shell
                full,
                input=payload,
                capture_output=True,
                env={**_host_path(), **NO_REPLACE},
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return Result(124, "", f"sandbox timed out after {timeout}s\n")
        except OSError as exc:
            return Result(127, "", f"cannot start docker: {exc}\n")
        out = res.stdout.decode("utf-8", "replace")
        err = res.stderr.decode("utf-8", "replace")
        return Result(res.returncode, out, err)


def _sh_quote(arg: str) -> str:
    return shlex.quote(arg)


def _host_path() -> dict[str, str]:
    keep = ("PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG")
    return {k: os.environ[k] for k in keep if k in os.environ}


def _container_env(env: Mapping[str, str], cwd: Path) -> dict[str, str]:
    """Return ``env`` with container paths only: host caches dropped, homes and temp in /tmp."""
    out = {k: _container_path(v, cwd) for k, v in env.items() if k not in _HOST_ONLY_ENV}
    out.update(_CONTAINER_ENV)
    out.setdefault("PYTHONPATH", "src:.")  # the slice's own code; the image installs no project
    return out


def _container_path(value: str, cwd: Path) -> str:
    """Return ``value`` as its /slice path when it names a path under ``cwd``, else unchanged."""
    if not value.startswith("/"):
        return value
    try:
        rel = Path(value).resolve().relative_to(cwd.resolve())
    except ValueError:
        return value
    return f"{WORKDIR}/{rel}" if str(rel) != "." else WORKDIR


def _container_argv(argv: Sequence[str], cwd: Path, extras: list[tuple[str, Path]]) -> list[str]:
    """Return ``argv`` for the container.

    A host interpreter in front becomes the image's ``python``; a path under ``cwd`` becomes its
    /slice path; another host file (the run's private pytest gate) is shipped in the slice tar
    under ``.sandbox/`` and named there.
    """
    out = []
    for i, arg in enumerate(argv):
        path = Path(arg)
        if i == 0 and path.is_absolute() and path.name.startswith("python"):
            out.append("python")
            continue
        translated = _container_path(arg, cwd)
        if translated != arg or not path.is_absolute() or not path.is_file():
            out.append(translated)
            continue
        arcname = f"{_EXTRAS}/{len(extras)}-{path.name}"
        extras.append((arcname, path))
        out.append(f"{WORKDIR}/{arcname}")
    return out


def _slice_tar(cwd: Path, extras: Sequence[tuple[str, Path]] = ()) -> bytes:
    """Return a tar of the working directory's files (no .git, no .venv) plus ``extras``."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for item in sorted(cwd.rglob("*")):
            rel = item.relative_to(cwd)
            if ".git" in rel.parts or any(p.startswith(".venv") for p in rel.parts):
                continue
            tar.add(item, arcname=str(rel), recursive=False)
        for arcname, path in extras:
            tar.add(path, arcname=arcname, recursive=False)
    return buf.getvalue()


def build_executor(cfg: Config, env_allow: Sequence[str]) -> DockerExecutor:
    """Return a :class:`DockerExecutor` configured from ``cfg``."""
    return DockerExecutor(
        cli=cfg.docker_cli,
        image=cfg.docker_image,
        dockerfile=cfg.repo / cfg.dockerfile,
        lock=cfg.repo / "uv.lock",
        env_allow=env_allow,
        memory=cfg.docker_memory,
        pids=cfg.docker_pids,
        build_timeout=cfg.docker_run_timeout_s,
        network=cfg.sandbox_network,
        neo4j_uri=cfg.sandbox_neo4j_uri,
        neo4j_user=cfg.sandbox_neo4j_user,
    )
