"""Timestamped, overwrite-on-run logging for Python workflow entry points."""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence, TextIO


def _stamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


class TimestampedTee:
    """Write each terminal line to both the original stream and a log file."""

    def __init__(self, terminal: TextIO, log_file: TextIO) -> None:
        self._terminal = terminal
        self._log_file = log_file
        self._at_line_start = True

    @property
    def encoding(self) -> str:
        return "utf-8"

    def write(self, text: str) -> int:
        if not text:
            return 0
        for piece in text.splitlines(keepends=True) or [text]:
            prefix = f"[{_stamp()}] " if self._at_line_start else ""
            rendered = prefix + piece
            self._terminal.write(rendered)
            self._log_file.write(rendered)
            self._at_line_start = piece.endswith(("\n", "\r"))
        self.flush()
        return len(text)

    def flush(self) -> None:
        self._terminal.flush()
        self._log_file.flush()


@contextmanager
def timestamped_log(project_root: Path, name: str) -> Iterator[TimestampedTee | None]:
    """Create a fixed log unless an outer project workflow already owns one."""

    if os.environ.get("QDE_LOG_ACTIVE") == "1":
        yield None
        return

    log_dir = Path(os.environ.get("QDE_LOG_DIR", str(project_root / "logs"))).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    prior_active = os.environ.get("QDE_LOG_ACTIVE")
    prior_file = os.environ.get("QDE_LOG_FILE")
    original_stdout, original_stderr = sys.stdout, sys.stderr

    with log_path.open("w", encoding="utf-8", buffering=1) as handle:
        tee = TimestampedTee(original_stdout, handle)
        os.environ["QDE_LOG_ACTIVE"] = "1"
        os.environ["QDE_LOG_FILE"] = str(log_path)
        sys.stdout = tee  # type: ignore[assignment]
        sys.stderr = tee  # type: ignore[assignment]
        try:
            print(f"Started {name}. Log file: {log_path}")
            yield tee
        except Exception:
            traceback.print_exc()
            raise
        finally:
            print(f"Finished {name}.")
            sys.stdout, sys.stderr = original_stdout, original_stderr
            if prior_active is None:
                os.environ.pop("QDE_LOG_ACTIVE", None)
            else:
                os.environ["QDE_LOG_ACTIVE"] = prior_active
            if prior_file is None:
                os.environ.pop("QDE_LOG_FILE", None)
            else:
                os.environ["QDE_LOG_FILE"] = prior_file


def run_logged_command(command: Sequence[str], sink: TimestampedTee | None) -> None:
    """Run a child command while preserving its timestamped output in the caller log."""

    if sink is None:
        subprocess.run(command, check=True)
        return
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        sink.write(line)
    process.stdout.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, list(command))
