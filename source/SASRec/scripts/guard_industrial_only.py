from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


BLOCKED = ("Musical_Instruments", "CDs_and_Vinyl")


def master_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: guard_industrial_only.py <master_pid>")
    master_pid = int(sys.argv[1])

    while master_alive(master_pid):
        output = subprocess.check_output(
            ["ps", "-eo", "pid=,args="],
            text=True,
        )
        for line in output.splitlines():
            if "--category" not in line:
                continue
            if not any(category in line for category in BLOCKED):
                continue
            pid_text, _, args = line.strip().partition(" ")
            pid = int(pid_text)
            if pid == os.getpid():
                continue
            print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} blocking {pid}: {args}", flush=True)
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        time.sleep(10)


if __name__ == "__main__":
    main()

