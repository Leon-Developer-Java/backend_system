from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Sequence


BACKEND_ROOT = Path(__file__).resolve().parents[1]


def start_adapter_worker(extra_args: Sequence[str] = ()) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    options: dict = {
        "cwd": BACKEND_ROOT,
        "env": env,
    }
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    return subprocess.Popen(
        [sys.executable, "-m", "workers.parse_worker", *extra_args],
        **options,
    )


def stop_adapter_worker(process: subprocess.Popen | None, timeout: float = 10) -> None:
    if process is None or process.poll() is not None:
        return

    if os.name == "nt":
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            process.send_signal(signal.CTRL_BREAK_EVENT)
            process.wait(timeout=timeout)
            return
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)

    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=timeout)
