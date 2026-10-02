#!/usr/bin/env python3
"""Mirror stdin to the terminal and a truncating, line-timestamped log file."""

from __future__ import annotations

import argparse
import codecs
import sys
from datetime import datetime
from pathlib import Path


def _stamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", required=True)
    args = parser.parse_args()
    log_file = Path(args.log_file).expanduser().resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # `w` intentionally replaces the prior transcript for this fixed workflow.
    with log_file.open("w", encoding="utf-8", buffering=1) as handle:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending = ""

        def emit(message: str) -> None:
            # tqdm and similar tools update a single terminal row using '\r'.
            # Normalize those updates into independent lines so that neither the
            # terminal nor the saved transcript can overwrite a timestamp.
            if not message:
                return
            stamped = f"[{_stamp()}] {message}\n"
            sys.stdout.write(stamped)
            sys.stdout.flush()
            handle.write(stamped)
            handle.flush()

        while chunk := sys.stdin.buffer.read1(8192):
            pending += decoder.decode(chunk)
            start = 0
            for index, character in enumerate(pending):
                if character in {"\r", "\n"}:
                    emit(pending[start:index])
                    start = index + 1
            pending = pending[start:]

        pending += decoder.decode(b"", final=True)
        emit(pending)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
