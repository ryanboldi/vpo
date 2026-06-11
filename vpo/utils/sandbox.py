"""Hardened sandboxed code execution for untrusted model output.

Inspired by HumanEval / LeetCodeDataset `execution.py`, with stronger
filesystem isolation. The threat model is *accidental* damage from
model-generated code (infinite loops, memory blowups, stray file writes,
fork bombs, sys.exit calls). It is NOT a security sandbox against an
adversarial attacker — that requires containers / seccomp / namespaces.

Defenses, in layers:

  1. Subprocess isolation — each call runs in a `spawn`-launched worker
     from a persistent ProcessPoolExecutor. Workers are reused across
     calls but killed-and-respawned on timeout.
  2. Kernel resource limits set in worker init:
       - RLIMIT_AS / RLIMIT_DATA / RLIMIT_STACK — virtual memory cap
       - RLIMIT_FSIZE = 0 — kernel blocks ALL writes to regular files,
         even via ctypes / C extensions. This is the strongest single
         defense against file creation.
  3. `reliability_guard` — nulls out destructive functions in os,
     shutil, subprocess, plus blocks dangerous module imports.
  4. `builtins.open` is replaced with a read-only wrapper that rejects
     write/append/exclusive/update modes at the Python level (RLIMIT_FSIZE
     backstops it at the kernel level).
  5. Each worker chdirs into a per-worker tmpdir (created BEFORE
     os.chdir is nulled out), so any path-relative file ops are scoped.
  6. Per-call `swallow_io` redirects stdout/stderr/stdin so user prints
     don't pollute logs and `input()` raises immediately.
  7. Two-layer timeout:
       - Inner: `signal.SIGALRM` raises `TimeoutException` inside the
         worker — fast, lets the worker keep running for the next call.
       - Outer: `future.result(timeout=...)` from the parent — backstops
         the case where user code installs its own SIGALRM handler or
         blocks signals. Triggers a hard pool reset (SIGKILL).

This module lives in `vpo.utils.sandbox` (a normal importable
package module) rather than inside mbpp_reward.py — veRL loads the
reward file via importlib.spec_from_file_location, which gives it a
synthetic `custom_module_NNN` name that cannot be re-imported in spawn
workers. Pickling functions from such a module fails with:
    PicklingError: Can't pickle <function ...>: import of module
    'custom_module_NNN' failed
"""

import atexit
import contextlib
import faulthandler
import io
import math
import multiprocessing
import os
import platform
import shutil
import signal
import sys
import tempfile
import threading
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
from concurrent.futures.process import BrokenProcessPool


# ─── Configuration ──────────────────────────────────────────────────────────

_WORKER_MEM_LIMIT = 8 * 1024 * 1024 * 1024  # 8 GB virtual address space
_WORKER_FSIZE_LIMIT = 0                      # bytes — block ALL file writes
_WORKER_COUNT = 4
_MP_CTX = multiprocessing.get_context("spawn")

# Parent-owned base dir holding every per-worker scratch dir. Created lazily
# before the pool starts and handed to spawn workers via env var (spawn workers
# inherit the parent environment). Removed at interpreter exit so a long run
# that resets the pool on every timeout doesn't leak `vpo_sbx_*` dirs into /tmp.
_SANDBOX_BASE: str | None = None
_SANDBOX_ATEXIT_REGISTERED = False


def _ensure_sandbox_base() -> str:
    global _SANDBOX_BASE, _SANDBOX_ATEXIT_REGISTERED
    if _SANDBOX_BASE is None or not os.path.isdir(_SANDBOX_BASE):
        _SANDBOX_BASE = tempfile.mkdtemp(prefix="vpo_sbx_base_")
        # Spawn workers inherit this env var and mkdtemp inside it.
        os.environ["VPO_SANDBOX_BASE"] = _SANDBOX_BASE
        if not _SANDBOX_ATEXIT_REGISTERED:  # register once, not per pool reset
            atexit.register(_cleanup_sandbox_base)
            _SANDBOX_ATEXIT_REGISTERED = True
    return _SANDBOX_BASE


def _cleanup_sandbox_base() -> None:
    # Operate on the module global (the parent's source of truth), not the env
    # var, and clear it so the next _ensure_sandbox_base recreates a live dir.
    global _SANDBOX_BASE
    base = _SANDBOX_BASE
    if base and os.path.isdir(base):
        shutil.rmtree(base, ignore_errors=True)
    _SANDBOX_BASE = None


# ─── I/O containment + timeout primitives (LeetCode/HumanEval style) ────────


class TimeoutException(Exception):
    pass


class _WriteOnlyStringIO(io.StringIO):
    """StringIO that raises on read so user code can't drain its own output."""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        return False


class _redirect_stdin(contextlib._RedirectStream):  # type: ignore[misc]
    _stream = "stdin"


@contextlib.contextmanager
def _swallow_io():
    stream = _WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with _redirect_stdin(stream):
                yield


@contextlib.contextmanager
def _time_limit(seconds: float):
    """SIGALRM-based wall-clock timeout. Best-effort: silently no-ops if
    we're not on the main thread (where signal handlers can be installed)."""
    def handler(signum, frame):
        raise TimeoutException("Sandboxed execution timed out")

    installed = False
    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        installed = True
    except (ValueError, OSError):
        # Not in main thread, or platform doesn't support setitimer.
        # The outer ProcessPoolExecutor timeout will still backstop us.
        pass
    try:
        yield
    finally:
        if installed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)


# ─── reliability_guard: disable destructive Python-level functions ──────────


def _reliability_guard():
    """Disable destructive functions in builtins, os, shutil, subprocess.

    Applied ONCE at worker startup. The pool worker only ever runs `exec()`
    of user code, so it never legitimately needs any of these functions.
    """
    import builtins

    # Read-only wrapper around `open`. RLIMIT_FSIZE=0 also blocks writes
    # at the kernel level, but rejecting at the Python level gives a clear
    # PermissionError instead of a SIGXFSZ-induced crash.
    _real_open = builtins.open

    def _ro_open(file, mode="r", *args, **kwargs):
        # Reject any mode that allows writing. Read-only modes are 'r', 'rb',
        # 'rt'. Anything containing w/a/x/+ writes.
        if isinstance(mode, str) and any(c in mode for c in ("w", "a", "x", "+")):
            raise PermissionError(
                f"sandboxed: open(mode={mode!r}) is blocked"
            )
        return _real_open(file, mode, *args, **kwargs)

    builtins.open = _ro_open  # type: ignore[assignment]
    builtins.exit = None      # type: ignore[assignment]
    builtins.quit = None      # type: ignore[assignment]
    builtins.help = None      # type: ignore[assignment]
    builtins.breakpoint = None  # type: ignore[assignment]
    # NOTE: builtins.input is intentionally NOT nulled. LiveCodeBench stdin
    # problems legitimately call input(); when running in MBPP-style mode
    # the swallow_io context wraps stdin in a non-readable StringIO so
    # input() will raise IOError immediately instead of blocking. In
    # safe_execute_stdin mode we provide a real readable StringIO.

    os.environ["OMP_NUM_THREADS"] = "1"

    # Wrap os.open to reject any flag that could create or modify a file.
    # `builtins.open` ultimately routes through this for file paths, but
    # user code can also call `os.open` directly.
    _real_os_open = os.open
    _write_flags = (
        os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    )

    def _ro_os_open(path, flags, mode=0o777, *, dir_fd=None):
        if flags & _write_flags:
            raise PermissionError(
                f"sandboxed: os.open(flags={flags!r}) is blocked"
            )
        return _real_os_open(path, flags, mode, dir_fd=dir_fd)

    os.open = _ro_os_open  # type: ignore[assignment]

    # Null out destructive os.* functions.
    for name in (
        "kill", "system", "putenv", "remove", "removedirs", "rmdir",
        "fchdir", "setuid", "fork", "forkpty", "killpg", "rename",
        "renames", "truncate", "replace", "unlink", "fchmod", "fchown",
        "chmod", "chown", "chroot", "lchflags", "lchmod", "lchown",
        "getcwd", "chdir",
    ):
        if hasattr(os, name):
            try:
                setattr(os, name, None)
            except (AttributeError, TypeError):
                pass

    import shutil
    for name in ("rmtree", "move", "chown", "copy", "copy2", "copyfile",
                 "copytree", "make_archive"):
        if hasattr(shutil, name):
            setattr(shutil, name, None)

    import subprocess
    subprocess.Popen = None        # type: ignore[assignment]
    subprocess.run = None          # type: ignore[assignment]
    subprocess.call = None         # type: ignore[assignment]
    subprocess.check_call = None   # type: ignore[assignment]
    subprocess.check_output = None  # type: ignore[assignment]

    # Block import of debug / process / GUI / FFI modules. Setting
    # sys.modules[m] to None makes future `import m` raise ImportError.
    # ctypes is the only realistic way for Python code to bypass our
    # builtins.open / os.open wrappers (it can call libc.open directly),
    # so blocking it is the last line of defense before kernel sandboxing.
    for mod in ("ctypes", "_ctypes", "cffi", "_cffi_backend",
                "ipdb", "pdb", "joblib", "psutil", "tkinter",
                "smtplib", "ftplib", "telnetlib", "webbrowser"):
        sys.modules[mod] = None  # type: ignore[assignment]


def _set_rlimits():
    """Apply kernel-enforced resource limits in the worker process."""
    try:
        import resource
        resource.setrlimit(
            resource.RLIMIT_AS, (_WORKER_MEM_LIMIT, _WORKER_MEM_LIMIT)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (_WORKER_MEM_LIMIT, _WORKER_MEM_LIMIT)
        )
        if platform.uname().system != "Darwin":
            try:
                resource.setrlimit(
                    resource.RLIMIT_STACK,
                    (_WORKER_MEM_LIMIT, _WORKER_MEM_LIMIT),
                )
            except (ValueError, OSError):
                pass
        # The strongest single defense against file creation: kernel
        # blocks any write that would make a regular file exceed 0 bytes.
        # Pipes / sockets / ttys are unaffected (so worker stdout still
        # works), but any open(path, 'w').write(...) gets SIGXFSZ.
        resource.setrlimit(
            resource.RLIMIT_FSIZE, (_WORKER_FSIZE_LIMIT, _WORKER_FSIZE_LIMIT)
        )
    except (ValueError, OSError):
        pass


# ─── Worker initializer + per-call entry point ──────────────────────────────


def _init_worker():
    """Subprocess initializer: set up the full sandbox once per worker."""
    # Don't write .pyc files — would otherwise try to materialize bytecode
    # caches under __pycache__ when user code triggers an import.
    sys.dont_write_bytecode = True

    faulthandler.disable()
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Create a per-worker tmpdir BEFORE reliability_guard nulls os.chdir.
    # Anything (accidentally) written by user code will land here.
    try:
        base = os.environ.get("VPO_SANDBOX_BASE") or None
        tmpdir = tempfile.mkdtemp(prefix="vpo_sbx_", dir=base)
        os.chdir(tmpdir)
    except OSError:
        pass

    _set_rlimits()
    _reliability_guard()


def _strip_main_blocks(code: str, strip_main_guard: bool = True) -> str:
    """Remove top-level process-terminating lines (unittest.main / sys.exit /
    exit() / quit()) so user code can't kill the worker mid-exec.

    When ``strip_main_guard`` is True (functional-test mode — the harness calls
    the target function directly, so the guard body is dead weight), the whole
    ``if __name__ == '__main__':`` block is dropped too. For stdin-style
    problems the *driver lives inside* that guard, so the caller passes
    ``strip_main_guard=False`` and instead sets ``__name__ == '__main__'`` in
    the exec globals so the guard runs naturally — otherwise the solution's
    real logic would be deleted and it would score 0.
    """
    lines = code.split("\n")
    out = []
    skip_indent = None
    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if skip_indent is not None:
            if stripped == "" or indent > skip_indent:
                continue
            skip_indent = None
        if strip_main_guard and stripped.startswith("if __name__"):
            skip_indent = indent
            continue
        # Only strip TOP-LEVEL (indent 0) terminators. An *indented*
        # sys.exit()/exit()/quit() is typically the sole body of a guard or
        # branch (`if bad: sys.exit()`); deleting it orphans the header
        # (IndentationError) or drops an early return — both fail a correct
        # program. Indented terminators can't kill the worker anyway: they
        # raise SystemExit, which _run_test_inner/_run_stdin_inner catch.
        if indent == 0 and stripped.startswith(
            ("unittest.main", "sys.exit", "exit(", "quit(")
        ):
            continue
        out.append(line)
    return "\n".join(out)


def _normalize_stdout(s: str) -> str:
    """Judge-style whitespace normalization for stdout comparison.

    Strips trailing whitespace on each line and drops trailing blank lines, so
    cosmetic differences (a trailing space per line, an extra final newline)
    don't fail an otherwise-correct answer — matching how AtCoder/Codeforces
    judges compare. A bare ``.strip()`` only trims the whole-string ends and so
    rejects correct multi-line output with per-line trailing whitespace.
    """
    lines = [ln.rstrip() for ln in s.replace("\r\n", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _run_test_inner(code: str, test_case: str, timeout: float) -> bool:
    """Run inside a sandboxed worker. Returns True iff the test passes."""
    cleaned = _strip_main_blocks(code)
    try:
        exec_globals = {"math": math, "__builtins__": __builtins__}
        with _swallow_io():
            with _time_limit(timeout):
                exec(cleaned, exec_globals)  # noqa: S102
                exec(test_case, exec_globals)  # noqa: S102
        return True
    except BaseException:
        return False


def _run_stdin_inner(
    code: str, stdin_input: str, expected_output: str, timeout: float
) -> bool:
    """Run inside a sandboxed worker for stdin-style problems.

    Wraps user code with:
      - sys.stdin = readable StringIO(stdin_input)
      - sys.stdout = capture StringIO
    Then compares stripped capture against stripped expected_output.

    Used by LiveCodeBench atcoder/codeforces problems where the model
    writes a complete script that reads stdin and prints to stdout.
    """
    # Keep the `if __name__ == '__main__':` block (stdin solutions put their
    # driver there) and make it run by setting __name__ in the globals.
    cleaned = _strip_main_blocks(code, strip_main_guard=False)
    try:
        stdin_buf = io.StringIO(stdin_input)
        stdout_buf = io.StringIO()
        exec_globals = {
            "math": math,
            "__builtins__": __builtins__,
            "__name__": "__main__",
        }
        # Use real (readable) stdin + real capture stdout — NOT _swallow_io,
        # which would block reads. The other sandbox defenses (rlimits,
        # reliability_guard, ctypes block, signal time_limit, future.result
        # backstop) are still in force from _init_worker.
        with contextlib.redirect_stdout(stdout_buf):
            with _redirect_stdin(stdin_buf):
                with _time_limit(timeout):
                    try:
                        exec(cleaned, exec_globals)  # noqa: S102
                    except SystemExit:
                        # sys.exit() after printing is normal completion, not a
                        # failure — the output is already captured.
                        pass
        return _normalize_stdout(stdout_buf.getvalue()) == _normalize_stdout(
            expected_output
        )
    except BaseException:
        return False


# ─── Pool management ────────────────────────────────────────────────────────


# Lazy-initialized so the executor is created in the worker that imports
# this module, not at module-load time.
_executor_lock = threading.Lock()
_executor: ProcessPoolExecutor | None = None

# Generous wait for the first worker to come up. Spawn workers re-import
# the parent's __main__ module, which can transitively import heavy libs
# (datasets, torch, vllm). Without pre-warming, the first user-facing
# safe_execute call would hit the regular `timeout + 1` outer deadline
# while workers are still importing — every call would time out, the pool
# would be killed, and the next call would face the same cold-start cycle.
_PREWARM_TIMEOUT = 90.0


def _noop_warmup() -> bool:
    """Top-level no-op so it can be pickled for the spawn pre-warm."""
    return True


def _spawn_executor() -> ProcessPoolExecutor:
    """Create a pool and force every worker through its cold start.

    We submit one no-op per slot at once so the pool is forced to create all
    workers in parallel; then wait for each. This pays the cold-start cost
    ONCE, at spawn, instead of having every early caller race the import
    against the ~7s outer deadline (each miss would kill the pool and start
    the cold-start cycle over).
    """
    executor = ProcessPoolExecutor(
        max_workers=_WORKER_COUNT,
        initializer=_init_worker,
        mp_context=_MP_CTX,
    )
    futures = [executor.submit(_noop_warmup) for _ in range(_WORKER_COUNT)]
    for f in futures:
        try:
            f.result(timeout=_PREWARM_TIMEOUT)
        except Exception:
            # Best effort — if a worker can't even import, the
            # next safe_execute will surface the real error.
            pass
    return executor


def _get_executor() -> ProcessPoolExecutor:
    global _executor
    with _executor_lock:
        if _executor is None:
            _ensure_sandbox_base()
            _executor = _spawn_executor()
        return _executor


def _reset_executor():
    """Recreate the pool, force-killing worker processes first.

    ProcessPoolExecutor.shutdown(wait=False) does NOT kill running workers,
    it just stops accepting new tasks. To stop a runaway infinite-loop user
    program (e.g. one that ignored our SIGALRM), we have to SIGKILL the
    worker processes directly.
    """
    global _executor
    with _executor_lock:
        if _executor is not None:
            try:
                workers = list(getattr(_executor, "_processes", {}).values())
            except Exception:
                workers = []
            for proc in workers:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                _executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        # The killed workers' scratch dirs are now orphaned — reclaim them
        # before the fresh pool spawns new ones, so resets don't accumulate.
        _cleanup_sandbox_base()
        _ensure_sandbox_base()
        # Pre-warm exactly like first spawn: without it, the next caller's
        # ~7s outer deadline races the cold spawn (re-importing the parent
        # __main__ with torch/vllm), times out spuriously, resets again —
        # cascading false 0.0 scores for the rest of the batch.
        _executor = _spawn_executor()


def safe_execute(code: str, test_case: str, timeout: int = 5) -> bool:
    """Execute `code` followed by `test_case` in a hardened sandbox.

    Returns True iff both exec calls succeed without raising. False on
    any exception (including timeout, memory limit, blocked operation,
    or simple test-case assertion failure).

    Two-layer timeout:
      1. Inner SIGALRM (`timeout` seconds) — raises TimeoutException
         inside the worker; the worker survives and serves the next call.
      2. Outer future.result (`timeout + 1` seconds) — backstops case
         where user code installed its own SIGALRM handler or blocks
         signals. Triggers a SIGKILL pool reset.
    """
    inner_timeout = float(timeout)
    outer_timeout = float(timeout) + 1.0
    try:
        future = _get_executor().submit(
            _run_test_inner, code, test_case, inner_timeout
        )
        return future.result(timeout=outer_timeout)
    except FuturesTimeout:
        _reset_executor()
        return False
    except BrokenProcessPool:
        _reset_executor()
        return False
    except Exception:
        return False


def safe_execute_stdin(
    code: str, stdin_input: str, expected_output: str, timeout: int = 6
) -> bool:
    """Execute `code` with `stdin_input` piped in; True iff stdout matches.

    Comparison is judge-style normalized (`_normalize_stdout`: per-line trailing
    whitespace stripped and trailing blank lines dropped on both sides). Used for
    LiveCodeBench atcoder/codeforces-style problems where the model writes
    a script that reads stdin and prints to stdout.

    Same two-layer timeout as `safe_execute`.
    """
    inner_timeout = float(timeout)
    outer_timeout = float(timeout) + 1.0
    try:
        future = _get_executor().submit(
            _run_stdin_inner, code, stdin_input, expected_output, inner_timeout
        )
        return future.result(timeout=outer_timeout)
    except FuturesTimeout:
        _reset_executor()
        return False
    except BrokenProcessPool:
        _reset_executor()
        return False
    except Exception:
        return False
