"""Stand-in child process for runner tests: prints numbered lines forever.

On SIGINT it prints CLEAN EXIT and exits 0 (mimicking the rollout scripts'
graceful shutdown); with --stubborn it ignores SIGINT so tests can exercise
the SIGKILL-after-grace path.
"""

import signal
import sys
import time


def main() -> int:
    if "--stubborn" in sys.argv:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    else:
        def clean_exit(_sig, _frame):
            print("CLEAN EXIT", flush=True)
            sys.exit(0)
        signal.signal(signal.SIGINT, clean_exit)

    print("READY", flush=True)
    i = 0
    while True:
        print(f"line {i}", flush=True)
        i += 1
        time.sleep(0.02)


if __name__ == "__main__":
    raise SystemExit(main())
